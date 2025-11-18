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

# DB_URL = "sqlite:///./data.db"
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db")
COUNTRY = "us"
CHART_TYPE = "top-grossing"
LIMIT = 100  
# токен для ручного обновления, можно переопределить через переменную окружения
FETCH_TOKEN = os.getenv("FETCH_TOKEN", "super_secret_token_change_me")
FETCH_INTERVAL_SECONDS = 24 * 60 * 60  # автообновление раз в день

# задержки, чтобы не упираться в лимиты RSS
CATEGORY_DELAY_SECONDS = float(os.getenv("CATEGORY_DELAY_SECONDS", "3"))  # пауза между категориями (сек)
RETRY_DELAYS = [2, 4, 8, 16, 30]  # увеличены задержки для retry

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

# Мультипликаторы трафика по категориям (относительно базовой категории)
CATEGORY_TRAFFIC_MULTIPLIER: Dict[int, float] = {
    6000: 1.5,   # Business
    6001: 0.4,   # Weather
    6002: 1.0,   # Utilities
    6003: 1.2,   # Travel
    6004: 1.0,   # Sports
    6005: 1.3,   # Social Networking
    6006: 0.7,   # Reference
    6007: 1.4,   # Productivity
    6008: 4.0,   # Photo & Video
    6009: 1.1,   # News
    6010: 1.0,   # Navigation
    6011: 3.0,   # Music
    6012: 1.2,   # Lifestyle
    6013: 2.0,   # Health & Fitness
    6015: 1.5,   # Finance
    6016: 2.5,   # Entertainment
    6017: 1.3,   # Education
    6018: 0.8,   # Books
    6020: 0.5,   # Medical
    6023: 0.2,   # Catalogs
    6024: 1.0,   # Food & Drink
    6026: 0.7,   # Kids
    6027: 0.6,   # Magazines & Newspapers
}

# Базовая выручка категории в день (USD) - для расчета коэффициентов
BASE_DAILY_INSTALLS = 100000  # Базовое количество установок для категории

# Модель монетизации (средние значения для top-grossing приложений)
REVENUE_MODEL = {
    "trial_conversion": 0.07,      # 7% пользователей начинают триал
    "trial_to_pay": 0.40,           # 40% переходят в платных
    "net_monthly_arpu": 12,         # $12 средний чек в месяц
    "ltv_multiplier": 7.0,          # LTV = 7 месяцев среднее удержание
}

# Кривая трафика по позициям (% от общего трафика категории)
def get_traffic_curve_coefficient(rank: int) -> float:
    """
    Возвращает коэффициент трафика в зависимости от позиции в топе.
    Использует бета-кривую распределения, похожую на реальную.
    """
    if rank == 1:
        return 0.10  # 10% трафика категории
    elif rank <= 5:
        return 0.06  # 6%
    elif rank <= 10:
        return 0.035  # 3.5%
    elif rank <= 20:
        return 0.020  # 2%
    elif rank <= 50:
        return 0.016  # 1.6%
    elif rank <= 100:
        return 0.012  # 1.2%
    else:
        return 0.005  # 0.5%


def estimate_daily_installs(rank: int, category_id: int) -> tuple[int, int]:
    """
    Оценивает диапазон ежедневных установок на основе позиции и категории.
    Возвращает (min_installs, max_installs).
    """
    traffic_coef = get_traffic_curve_coefficient(rank)
    category_mult = CATEGORY_TRAFFIC_MULTIPLIER.get(category_id, 1.0)
    
    base_installs = BASE_DAILY_INSTALLS * traffic_coef * category_mult
    
    # Добавляем погрешность ±30%
    min_installs = int(base_installs * 0.7)
    max_installs = int(base_installs * 1.3)
    
    return min_installs, max_installs


def estimate_monthly_revenue(daily_installs_avg: int) -> tuple[int, int]:
    """
    Оценивает месячную выручку на основе среднего количества установок в день.
    Возвращает (min_revenue, max_revenue) в USD.
    """
    # Базовый расчет
    paying_users = (
        daily_installs_avg 
        * REVENUE_MODEL["trial_conversion"] 
        * REVENUE_MODEL["trial_to_pay"]
    )
    
    monthly_revenue = (
        paying_users 
        * REVENUE_MODEL["net_monthly_arpu"] 
        * 30  # дней в месяце
    )
    
    # Добавляем погрешность ±40% (монетизация варьируется сильнее)
    min_revenue = int(monthly_revenue * 0.6)
    max_revenue = int(monthly_revenue * 1.4)
    
    return min_revenue, max_revenue


def calculate_position_change_impact(rank_from: Optional[int], rank_to: int) -> dict:
    """
    Рассчитывает влияние изменения позиции на метрики.
    """
    if rank_from is None:
        return {
            "change_percent": None,
            "traffic_impact": "NEW entry",
        }
    
    traffic_from = get_traffic_curve_coefficient(rank_from)
    traffic_to = get_traffic_curve_coefficient(rank_to)
    
    if traffic_from > 0:
        change_percent = ((traffic_to - traffic_from) / traffic_from) * 100
    else:
        change_percent = 0
    
    return {
        "change_percent": round(change_percent, 1),
        "traffic_impact": f"{'↑' if change_percent > 0 else '↓'}{abs(change_percent):.1f}% traffic",
    }


# список user-agent'ов, чтобы не светиться одним и тем же
USER_AGENTS: List[str] = [
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
    release_date = Column(String, nullable=True)  # Дата релиза приложения


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


def fetch_with_rotation(url: str, timeout: int = 15) -> requests.Response:
    """
    Делает GET-запрос к RSS API с ротацией user-agent и повторными попытками при временных ошибках.
    """
    last_exc: Optional[Exception] = None
    attempts = len(RETRY_DELAYS)

    for attempt in range(attempts):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json,text/*;q=0.9,*/*;q=0.8",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            print(f"[FETCH ERROR] network failure for {url}: {e}")
        else:
            if resp.status_code == 503:
                last_exc = requests.HTTPError(f"503 from {url}")
            else:
                try:
                    resp.raise_for_status()
                except requests.HTTPError as e:
                    last_exc = e
                else:
                    return resp

        if attempt < attempts - 1:
            time.sleep(RETRY_DELAYS[attempt])

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("fetch_with_rotation: exhausted attempts without response")



def fetch_category_rss(category_id: int) -> List[dict]:
    """
    Забираем RSS топ‑гроссинга для конкретной категории.
    """
    # Используем старый iTunes RSS формат, который еще работает
    url = (
        f"https://itunes.apple.com/{COUNTRY}/rss/topgrossingapplications/"
        f"limit={LIMIT}/genre={category_id}/json"
    )
    resp = fetch_with_rotation(url, timeout=60)
    data = resp.json()
    
    # Старый iTunes RSS использует структуру feed.entry вместо feed.results
    entries = data.get("feed", {}).get("entry", [])
    items: List[dict] = []

    for idx, entry in enumerate(entries, start=1):
        # Структура данных в старом API отличается
        # Безопасное получение данных с учетом что некоторые поля могут быть None или списками
        id_data = entry.get("id")
        name_data = entry.get("im:name")
        link_data = entry.get("link")
        release_data = entry.get("im:releaseDate")
        
        bundle_id = None
        if isinstance(id_data, dict):
            bundle_id = id_data.get("attributes", {}).get("im:bundleId")
        
        name = None
        if isinstance(name_data, dict):
            name = name_data.get("label")
        
        store_url = None
        if isinstance(link_data, dict):
            store_url = link_data.get("attributes", {}).get("href")
        if not store_url and isinstance(id_data, dict):
            store_url = id_data.get("label")
        
        release_date = None
        if isinstance(release_data, dict):
            release_date_str = release_data.get("label")
            if release_date_str:
                # Форматируем дату из ISO формата в простой вид (YYYY-MM-DD)
                try:
                    release_date = release_date_str.split("T")[0]
                except Exception:
                    release_date = release_date_str

        # попробуем взять имя категории из фида, если оно есть
        cat_name = NON_GAMING_CATEGORY_IDS.get(category_id) or ""
        if not cat_name:
            category_data = entry.get("category", {})
            if isinstance(category_data, dict):
                cat_name = category_data.get("attributes", {}).get("label", "")

        items.append(
            {
                "rank": idx,
                "bundle_id": bundle_id,
                "name": name,
                "store_url": store_url,
                "category_id": category_id,
                "category_name": cat_name or str(category_id),
                "release_date": release_date,
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
                    release_date=entry.get("release_date"),
                )
                db.add(app)
                db.flush()  # чтобы у app уже был id
            else:
                # Обновляем release_date если она пришла и еще не была установлена
                if entry.get("release_date") and not app.release_date:
                    app.release_date = entry.get("release_date")

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
    category: Optional[str] = Query(None),
    min_jump: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    sort_by: Optional[str] = Query(None),
    only_new: Optional[str] = Query(None),
    recent_days: Optional[str] = Query(None),
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
                "sort_by": None,
                "only_new": None,
                "recent_days": None,
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

    # разбор category (может прийти пустой строкой при "All categories")
    category_int: Optional[int] = None
    if category not in (None, ""):
        try:
            category_int = int(category)
        except ValueError:
            category_int = None

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

    if category_int is not None:
        snaps = snaps.filter(ChartSnapshot.category_id == category_int)

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
                "category_id": snap.category_id,
                "release_date": app.release_date,
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

        # Рассчитываем аналитику
        category_id = d.get("category_id")
        min_installs, max_installs = estimate_daily_installs(today_pos, category_id)
        avg_installs = (min_installs + max_installs) // 2
        min_revenue, max_revenue = estimate_monthly_revenue(avg_installs)
        
        position_impact = calculate_position_change_impact(prev_pos, today_pos)

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
                "release_date": d.get("release_date"),
                # Аналитика
                "est_installs_min": min_installs,
                "est_installs_max": max_installs,
                "est_revenue_min": min_revenue,
                "est_revenue_max": max_revenue,
                "traffic_impact": position_impact.get("traffic_impact"),
            }
        )

    # фильтр по "скачку" позиций (минимальный delta, только вверх)
    if min_jump_int is not None:
        rows = [
            r
            for r in rows
            if r.get("delta") is not None and r["delta"] >= min_jump_int
        ]

    # фильтр "только NEW"
    if only_new == "true":
        rows = [r for r in rows if r["status"] == "NEW"]

    # фильтр по недавно вышедшим приложениям
    recent_days_int: Optional[int] = None
    if recent_days not in (None, ""):
        try:
            recent_days_int = int(recent_days)
        except ValueError:
            recent_days_int = None
    
    if recent_days_int is not None:
        cutoff_date = (datetime.date.today() - datetime.timedelta(days=recent_days_int)).isoformat()
        rows = [
            r for r in rows 
            if r.get("release_date") and r["release_date"] >= cutoff_date
        ]

    # сортируем
    if sort_by == "new":
        # NEW приложения вверху, затем по дельте (большие скачки), затем по позиции
        rows.sort(key=lambda r: (
            0 if r["status"] == "NEW" else 1,
            -(r["delta"] if r["delta"] is not None else -999),
            r["today_pos"]
        ))
    elif sort_by == "delta":
        # Сортировка по дельте (большие скачки вверху)
        rows.sort(key=lambda r: (
            -(r["delta"] if r["delta"] is not None else -999),
            r["today_pos"]
        ))
    elif sort_by == "release":
        # Сортировка по дате релиза (новые вверху)
        rows.sort(key=lambda r: (
            r["release_date"] if r["release_date"] else "0000-00-00",  # Пустые даты вниз
        ), reverse=True)
    else:
        # По умолчанию сортируем по текущей позиции
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
            "selected_category": category_int,
            "min_jump": min_jump_int,
            "from_date": from_dt.isoformat() if from_dt else None,
            "to_date": to_dt.isoformat() if to_dt else None,
            "sort_by": sort_by,
            "only_new": only_new,
            "recent_days": recent_days_int,
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
