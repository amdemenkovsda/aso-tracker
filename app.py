import datetime
import asyncio
import os
import random
import secrets
import time
from typing import List, Optional, Dict, Generator

import requests
from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Date,
    ForeignKey,
    select,
    and_,
    delete,
    UniqueConstraint,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# --------------------------------------------------------------------
# Конфиг
# --------------------------------------------------------------------

DB_URL = "sqlite:///./data.db"
COUNTRY = "us"
CHART_TYPE = "top-grossing"
LIMIT = 200
# токен для ручного обновления, можно переопределить через переменную окружения
FETCH_TOKEN = os.getenv("FETCH_TOKEN", "super_secret_token_change_me")
FETCH_INTERVAL_SECONDS = 24 * 60 * 60  # автообновление раз в день

# задержки, чтобы не упираться в лимиты RSS
REQUEST_RETRY_DELAY = float(os.getenv("REQUEST_RETRY_DELAY", "2"))  # базовая пауза между ретраями одного запроса
REQUEST_MAX_ATTEMPTS = int(os.getenv("REQUEST_MAX_ATTEMPTS", "6"))  # сколько раз пытаемся сходить за RSS
REQUEST_BACKOFF_MULTIPLIER = float(os.getenv("REQUEST_BACKOFF_MULTIPLIER", "1.5"))  # как растёт задержка между попытками
REQUEST_JITTER_SECONDS = float(os.getenv("REQUEST_JITTER_SECONDS", "0.5"))  # случайный шум к задержке, чтобы не бомбить по расписанию
CATEGORY_DELAY_SECONDS = float(os.getenv("CATEGORY_DELAY_SECONDS", "1"))  # пауза между категориями (сек)
RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

# учётка для админ-доступа (можно переопределить через переменные окружения)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "aso")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "010203")

security = HTTPBasic()

NON_GAMING_CATEGORY_IDS: Dict[int, str] = {
    6000: "Business",
    6001: "Weather",
    6002: "Utilities",
    6003: "Travel",
    6004: "Sports",
    6005: "Social Networking",
    6006: "Reference",
    6007: "Productivity",
    6008: "Photo & Video",
    6009: "News",
    6010: "Navigation",
    6011: "Music",
    6012: "Lifestyle",
    6013: "Health & Fitness",
    6015: "Finance",
    6016: "Entertainment",
    6017: "Education",
    6018: "Books",
    6020: "Medical",
    6023: "Catalogs",
    6024: "Food & Drink",
    6026: "Kids",
    6027: "Magazines & Newspapers",
}

# список user-agent'ов, чтобы не светиться одним и тем же
USER_AGENTS: List[str] = [
    # можно добавить свои
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
]

# --------------------------------------------------------------------
# БД / ORM
# --------------------------------------------------------------------

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class AppModel(Base):
    __tablename__ = "apps"

    id = Column(Integer, primary_key=True, index=True)
    bundle_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    store_url = Column(String, nullable=True)


class ChartSnapshot(Base):
    __tablename__ = "chart_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(Date, index=True, nullable=False)
    country = Column(String, nullable=False)
    chart_type = Column(String, nullable=False)
    category_id = Column(Integer, nullable=False)
    category_name = Column(String, nullable=False)
    rank = Column(Integer, nullable=False)

    app_id = Column(Integer, ForeignKey("apps.id"), nullable=False)
    app = relationship("AppModel")

    __table_args__ = (
        UniqueConstraint(
            "snapshot_date",
            "country",
            "chart_type",
            "category_id",
            "rank",
            name="uq_snapshot_unique_row",
        ),
    )


Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------
# Аутентификация
# --------------------------------------------------------------------

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Простейшая HTTP Basic-авторизация для входа в админку.
    """
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------

request_session = requests.Session()


def _sleep_with_backoff(attempt: int) -> None:
    """
    Засыпаем с экспоненциальным ростом задержки и небольшим шумом,
    чтобы немного растянуть поток запросов.
    """
    delay = REQUEST_RETRY_DELAY * (REQUEST_BACKOFF_MULTIPLIER**attempt)
    if REQUEST_JITTER_SECONDS > 0:
        delay += random.uniform(0, REQUEST_JITTER_SECONDS)
    if delay > 0:
        time.sleep(delay)


def fetch_with_rotation(url: str, timeout: int = 15) -> requests.Response:
    """
    Делает GET-запрос с ротацией user-agent и ретраями без прокси.

    Логика:
    - количество попыток и задержки настраиваются через env;
    - на каждой попытке случайный User-Agent;
    - если сеть упала или получили код из списка RETRYABLE_STATUS_CODES —
      ждём с экспоненциальным ростом паузы и пробуем ещё раз;
    - для остальных HTTP-ошибок выкидываем исключение сразу.
    """
    last_exc: Optional[Exception] = None
    attempts = max(1, REQUEST_MAX_ATTEMPTS)

    for attempt in range(attempts):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json,text/*;q=0.9,*/*;q=0.8",
        }

        try:
            resp = request_session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            _sleep_with_backoff(attempt)
            continue
        if resp.status_code in RETRYABLE_STATUS_CODES:
            last_exc = requests.HTTPError(f"{resp.status_code} from {url}")
            _sleep_with_backoff(attempt)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            last_exc = e
            break

        return resp

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("fetch_with_rotation: exhausted attempts without response")



def fetch_category_rss(category_id: int) -> List[dict]:
    """
    Забираем RSS топ‑гроссинга для конкретной категории.
    """
    url = (
        f"https://rss.itunes.apple.com/api/v1/"
        f"{COUNTRY}/ios-apps/{CHART_TYPE}/all/{LIMIT}/explicit.json"
        f"?genre={category_id}"
    )
    resp = fetch_with_rotation(url, timeout=15)
    data = resp.json()
    results = data.get("feed", {}).get("results", [])
    items: List[dict] = []

    for idx, item in enumerate(results, start=1):
        bundle_id = item.get("bundleId")
        name = item.get("name")
        store_url = item.get("url")

        # попробуем взять имя категории из фида, если оно есть
        cat_name = NON_GAMING_CATEGORY_IDS.get(category_id) or ""
        if not cat_name:
            genres = item.get("genres") or []
            if genres:
                cat_name = genres[0].get("name") or ""

        items.append(
            {
                "rank": idx,
                "bundle_id": bundle_id,
                "name": name,
                "store_url": store_url,
                "category_id": category_id,
                "category_name": cat_name or str(category_id),
            }
        )

    return items


def fetch_and_store_all(db: Session, snapshot_date: datetime.date) -> int:
    """
    Проходим по всем non‑gaming категориям и сохраняем:
    - AppModel (справочник приложений)
    - ChartSnapshot (позиции по дням/категориям)
    """
    inserted = 0

    for idx, (cat_id, cat_name) in enumerate(NON_GAMING_CATEGORY_IDS.items()):
        # небольшая пауза между категориями, чтобы не долбить API слишком быстро
        if idx > 0 and CATEGORY_DELAY_SECONDS > 0:
            time.sleep(CATEGORY_DELAY_SECONDS)

        # пропускаем игрy (если что‑то такое попадётся)
        if "Game" in (cat_name or ""):
            continue

        try:
            entries = fetch_category_rss(cat_id)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] fetch category {cat_id}: {e}")
            continue

        for entry in entries:
            if not entry["bundle_id"] or not entry["name"]:
                continue

            app = (
                db.query(AppModel)
                .filter(AppModel.bundle_id == entry["bundle_id"])
                .first()
            )
            if app is None:
                app = AppModel(
                    bundle_id=entry["bundle_id"],
                    name=entry["name"],
                    store_url=entry.get("store_url"),
                )
                db.add(app)
                db.flush()  # чтобы у app уже был id

            # Variant B: не трогаем старые дни, а для конкретной даты делаем upsert по ключу
            existing = (
                db.query(ChartSnapshot)
                .filter(
                    ChartSnapshot.snapshot_date == snapshot_date,
                    ChartSnapshot.country == COUNTRY,
                    ChartSnapshot.chart_type == CHART_TYPE,
                    ChartSnapshot.category_id == entry["category_id"],
                    ChartSnapshot.rank == entry["rank"],
                )
                .first()
            )

            if existing is not None:
                # обновляем привязку к приложению и имя категории
                existing.app_id = app.id
                existing.category_name = entry["category_name"]
            else:
                snap = ChartSnapshot(
                    snapshot_date=snapshot_date,
                    country=COUNTRY,
                    chart_type=CHART_TYPE,
                    category_id=entry["category_id"],
                    category_name=entry["category_name"],
                    rank=entry["rank"],
                    app_id=app.id,
                )
                db.add(snap)
                inserted += 1

        db.commit()

    return inserted


# --------------------------------------------------------------------
# Вьюхи
# --------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    category: Optional[int] = Query(None),
    min_jump: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Главная таблица:
    - по умолчанию сравниваем последний день с предыдущим
    - можно выбирать категорию
    - можно фильтровать по "скачку" позиций (min_jump)
    - можно задавать диапазон дат
    """

    # последние две даты в базе
    dates_q = (
        db.query(ChartSnapshot.snapshot_date)
        .distinct()
        .order_by(ChartSnapshot.snapshot_date.desc())
    )
    dates = [row[0] for row in dates_q.all()]

    if not dates:
        # нет данных вообще
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "rows": [],
                "today": None,
                "compare_date": None,
                "chart_type": CHART_TYPE,
                "country": COUNTRY.upper(),
                "categories": [],
                "selected_category": None,
                "min_jump": None,
                "from_date": None,
                "to_date": None,
            },
        )

    latest_date = dates[0]
    prev_date = dates[1] if len(dates) > 1 else None

    # разбор дат из query
    def parse_date(value: Optional[str]) -> Optional[datetime.date]:
        if not value:
            return None
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            return None

    from_dt = parse_date(from_date) or prev_date or latest_date
    to_dt = parse_date(to_date) or latest_date

    # гарантируем, что from_dt не позже to_dt
    if from_dt > to_dt:
        from_dt, to_dt = to_dt, from_dt

    # разбор min_jump (может прийти пустой строкой)
    min_jump_int: Optional[int] = None
    if min_jump not in (None, ""):
        try:
            min_jump_int = int(min_jump)
        except ValueError:
            min_jump_int = None

    # список категорий, которые у нас есть в данных
    cat_rows = (
        db.query(ChartSnapshot.category_id, ChartSnapshot.category_name)
        .distinct()
        .order_by(ChartSnapshot.category_name)
        .all()
    )
    categories = [(cid, cname) for cid, cname in cat_rows]

    # достаём все срезы в диапазоне двух дат
    snaps = (
        db.query(ChartSnapshot, AppModel)
        .join(AppModel, ChartSnapshot.app_id == AppModel.id)
        .filter(
            ChartSnapshot.country == COUNTRY,
            ChartSnapshot.chart_type == CHART_TYPE,
            ChartSnapshot.snapshot_date.in_([from_dt, to_dt]),
        )
    )

    if category is not None:
        snaps = snaps.filter(ChartSnapshot.category_id == category)

    snaps = snaps.all()

    # собираем позиции "до" и "после" по каждому приложению
    data_by_app: Dict[int, dict] = {}

    for snap, app in snaps:
        d = data_by_app.setdefault(
            app.id,
            {
                "name": app.name,
                "bundle_id": app.bundle_id,
                "store_url": app.store_url,
                "category_name": snap.category_name,
                "pos_from": None,
                "pos_to": None,
            },
        )
        if snap.snapshot_date == from_dt:
            d["pos_from"] = snap.rank
        if snap.snapshot_date == to_dt:
            d["pos_to"] = snap.rank

    rows: List[dict] = []
    for d in data_by_app.values():
        today_pos = d["pos_to"]
        prev_pos = d["pos_from"]
        if today_pos is None:
            # приложение не попало в конечный день — игнорируем в этой таблице
            continue

        delta = None
        status = ""
        if prev_pos is None:
            status = "NEW"
        else:
            delta = prev_pos - today_pos
            if delta > 0:
                status = f"+{delta}"
            elif delta < 0:
                status = str(delta)
            else:
                status = "="

        rows.append(
            {
                "name": d["name"],
                "bundle_id": d["bundle_id"],
                "category_name": d["category_name"],
                "today_pos": today_pos,
                "prev_pos": prev_pos,
                "delta": delta,
                "status": status,
                "store_url": d["store_url"],
            }
        )

    # фильтр по "скачку" позиций (минимальный delta, только вверх)
    if min_jump_int is not None:
        rows = [
            r
            for r in rows
            if r.get("delta") is not None and r["delta"] >= min_jump_int
        ]

    # сортируем по текущей позиции
    rows.sort(key=lambda r: r["today_pos"])

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rows": rows,
            "today": to_dt,
            "compare_date": from_dt,
            "chart_type": CHART_TYPE,
            "country": COUNTRY.upper(),
            "categories": categories,
            "selected_category": category,
            "min_jump": min_jump_int,
            "from_date": from_dt.isoformat() if from_dt else None,
            "to_date": to_dt.isoformat() if to_dt else None,
        },
    )


@app.get("/fetch-now")
def fetch_now(
    token: str = Query(...),
    redirect: bool = Query(False),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if token != FETCH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    today = datetime.date.today()
    inserted = fetch_and_store_all(db, today)

    if redirect:
        return RedirectResponse(url="/", status_code=303)

    return {"status": "ok", "inserted": inserted}


# --------------------------------------------------------------------
# Автообновление по расписанию
# --------------------------------------------------------------------


@app.on_event("startup")
async def schedule_auto_fetch() -> None:
    async def worker() -> None:
        while True:
            # ждём до следующего запуска (раз в сутки)
            await asyncio.sleep(FETCH_INTERVAL_SECONDS)

            db = SessionLocal()
            try:
                today = datetime.date.today()
                inserted = fetch_and_store_all(db, today)
                print(f"[AUTO-FETCH] {today}: inserted {inserted} rows")
            except Exception as e:  # noqa: BLE001
                print(f"[AUTO-FETCH ERROR] {e}")
            finally:
                db.close()

    # запускаем фонового воркера
    asyncio.create_task(worker())
