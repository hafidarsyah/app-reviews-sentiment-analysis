import os
import csv
import uuid
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google_play_scraper import Sort, reviews_all, reviews
from google_play_scraper.constants.regex import Regex
import pandas as pd

# KONFIGURASI LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("scraping.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# KONFIGURASI GLOBAL
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset-scraping.csv")
COUNTRY = "id"
LANGUAGE = "id"
MAX_WORKERS = 8
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Daftar Aplikasi
TARGET_APPS = [
    {"app_id": "com.shopee.id", "app_name": "Shopee"},
]


# FUNGSI UTILITAS
def ensure_output_dir() -> None:
    """Membuat direktori output jika belum ada."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Direktori output siap: {OUTPUT_DIR}")


def init_csv(file_path: str) -> None:
    """Inisialisasi file CSV dengan header jika belum ada."""
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "app_id", "app_name", "review_id", "review_text",
                "rating", "review_date", "scraped_at"
            ])
        logger.info(f"File CSV dibuat: {file_path}")


def append_to_csv(file_path: str, rows: List[List], lock: Lock) -> None:
    """Menambahkan baris ke CSV secara thread-safe."""
    with lock:
        with open(file_path, "a", newline="", encoding="utf-8", errors="ignore") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def clean_review_text(text: Optional[str]) -> str:
    """Membersihkan teks ulasan dari karakter tidak valid."""
    if not text:
        return ""
    text = str(text).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def format_review_date(dt: Optional[datetime]) -> str:
    """Format objek datetime ke string YYYY-MM-DD HH:MM:SS."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# FUNGSI SCRAPING
def scrape_app_reviews(app: Dict, lock: Lock, file_path: str) -> int:
    """
    Scraping semua review dari satu aplikasi.
    Menggunakan reviews_all() untuk mengambil seluruh ulasan yang tersedia.

    Returns:
        Jumlah review yang berhasil di-scrape.
    """
    app_id = app["app_id"]
    app_name = app["app_name"]
    scraped_count = 0
    scraped_at = datetime.now().isoformat()

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"[{app_name}] Memulai scraping (percobaan {attempt}/{RETRY_ATTEMPTS})...")
            start_time = time.time()

            all_reviews = reviews_all(
                app_id,
                sleep_milliseconds=200,
                lang=LANGUAGE,
                country=COUNTRY,
                sort=Sort.MOST_RELEVANT
            )

            if not all_reviews:
                logger.warning(f"[{app_name}] Tidak ada review yang ditemukan.")
                return 0

            rows = []
            for r in all_reviews:
                review_text = clean_review_text(r.get("content"))
                if not review_text or len(review_text) < 3:
                    continue

                rows.append([
                    app_id,
                    app_name,
                    str(uuid.uuid4()),
                    review_text,
                    int(r.get("score", 0)),
                    format_review_date(r.get("at")),
                    scraped_at
                ])

            if rows:
                append_to_csv(file_path, rows, lock)
                scraped_count = len(rows)

            elapsed = time.time() - start_time
            logger.info(
                f"[{app_name}] Berhasil scrape {scraped_count} review "
                f"dalam {elapsed:.2f} detik."
            )
            return scraped_count

        except Exception as e:
            logger.error(f"[{app_name}] Error pada percobaan {attempt}: {e}")
            if attempt < RETRY_ATTEMPTS:
                logger.info(f"[{app_name}] Retry dalam {RETRY_DELAY} detik...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"[{app_name}] Gagal setelah {RETRY_ATTEMPTS} percobaan.")
                return 0


def scrape_app_paginated(
    app: Dict,
    lock: Lock,
    file_path: str,
    target_count: int = 1500
) -> int:
    """
    Scraping review dengan pagination untuk mengontrol jumlah data per aplikasi.
    """
    app_id = app["app_id"]
    app_name = app["app_name"]
    scraped_count = 0
    scraped_at = datetime.now().isoformat()
    continuation_token = None

    batch_size = 200
    batches_needed = (target_count + batch_size - 1) // batch_size

    logger.info(f"[{app_name}] Target: {target_count} review ({batches_needed} batch)")

    for batch_idx in range(batches_needed):
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                result, continuation_token = reviews(
                    app_id,
                    lang=LANGUAGE,
                    country=COUNTRY,
                    sort=Sort.MOST_RELEVANT,
                    count=min(batch_size, target_count - scraped_count),
                    continuation_token=continuation_token
                )

                if not result:
                    logger.info(f"[{app_name}] Tidak ada review lagi di batch {batch_idx + 1}")
                    return scraped_count

                rows = []
                for r in result:
                    review_text = clean_review_text(r.get("content"))
                    if not review_text or len(review_text) < 3:
                        continue

                    rows.append([
                        app_id,
                        app_name,
                        str(uuid.uuid4()),
                        review_text,
                        int(r.get("score", 0)),
                        format_review_date(r.get("at")),
                        scraped_at
                    ])

                if rows:
                    append_to_csv(file_path, rows, lock)
                    scraped_count += len(rows)

                logger.info(
                    f"[{app_name}] Batch {batch_idx + 1}/{batches_needed}: "
                    f"+{len(rows)} review (total: {scraped_count})"
                )

                if continuation_token is None:
                    logger.info(f"[{app_name}] Tidak ada halaman berikutnya.")
                    return scraped_count

                time.sleep(0.3)
                break

            except Exception as e:
                logger.error(
                    f"[{app_name}] Error batch {batch_idx + 1} "
                    f"(percobaan {attempt}): {e}"
                )
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.warning(
                        f"[{app_name}] Melewati batch {batch_idx + 1} setelah gagal."
                    )

    return scraped_count


# ORKESTRASI
def run_scraping(use_paginated: bool = True, target_per_app: int = 1500) -> None:
    """Menjalankan proses scraping untuk semua aplikasi target."""
    ensure_output_dir()
    init_csv(OUTPUT_FILE)

    lock = Lock()
    total_reviews = 0
    start_total = time.time()

    logger.info("=" * 60)
    logger.info("MEMULAI PROSES SCRAPING")
    logger.info(f"Target: {len(TARGET_APPS)} aplikasi, "
                f"{target_per_app} review/app (paginasi={use_paginated})")
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info("=" * 60)

    scraper_func = scrape_app_paginated if use_paginated else scrape_app_reviews

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                scraper_func, app, lock, OUTPUT_FILE, target_per_app
            ): app["app_name"]
            for app in TARGET_APPS
        }

        for future in as_completed(futures):
            app_name = futures[future]
            try:
                count = future.result()
                total_reviews += count
                logger.info(f"[{app_name}] Selesai: {count} review")
            except Exception as e:
                logger.error(f"[{app_name}] Exception: {e}")

    elapsed_total = time.time() - start_total
    logger.info("=" * 60)
    logger.info(f"PROSES SELESAI dalam {elapsed_total:.2f} detik")
    logger.info(f"Total review terkumpul: {total_reviews}")
    logger.info(f"Disimpan di: {OUTPUT_FILE}")
    logger.info("=" * 60)

    verify_dataset()


def verify_dataset() -> None:
    """Verifikasi dataset hasil scraping."""
    if not os.path.exists(OUTPUT_FILE):
        logger.warning("File dataset tidak ditemukan.")
        return

    try:
        df = pd.read_csv(OUTPUT_FILE)
        logger.info(f"Ukuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")
        logger.info(f"Distribusi rating:\n{df['rating'].value_counts().sort_index()}")
        logger.info(f"Distribusi aplikasi:\n{df['app_name'].value_counts()}")

        if df.shape[0] < 3000:
            logger.warning(
                f"Dataset kurang dari 3000 sampel ({df.shape[0]}). "
                "Tambah aplikasi atau tingkatkan target_per_app."
            )
        elif df.shape[0] < 10000:
            logger.info(
                f"Dataset sudah >3000, tapi disarankan >=10000 untuk akurasi optimal."
            )
        else:
            logger.info("Dataset memenuhi kriteria optimal (>=10000 sampel).")
    except Exception as e:
        logger.error(f"Gagal verifikasi dataset: {e}")


# ENTRY POINT
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Play Store Review Scraper"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "paginated"],
        default="paginated",
        help="Mode scraping: 'all' (semua review) atau 'paginated' (terkontrol)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1500,
        help="Target jumlah review per aplikasi (mode paginated)"
    )
    args = parser.parse_args()

    run_scraping(
        use_paginated=(args.mode == "paginated"),
        target_per_app=args.target
    )


if __name__ == "__main__":
    main()
