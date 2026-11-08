import re
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
BG = (18, 18, 20)
DETAIL_SIZE = 512   # размер HD-превью, которое подгружается при приближении
CYR = re.compile(r"[а-яё]", re.I)

_NSFW_RE = re.compile(
    r"\b(nude|naked|nudity|sex|sexual|sexuality|porn|porno|pornography|"
    r"erotic|erotica|penis|vagina|anus|anal|oral|breast\w*|nipple\w*|"
    r"cum|cumshot|semen|sperm|masturbat\w*|bdsm|bondage|ahegao|"
    r"lingerie|topless|explicit|hentai|penetrat\w*|futanari|femboy|"
    r"shota|testicle\w*|ass\b|asses\b|boobs\b|crotch|genital\w*|orgasm|fetish|"
    r"nsfw|creampie|deepthroat|handjob|blowjob|latex|swinger|orgy|"
    r"striptease)\b", re.I
)
# "non-sexual" — это SFW-якорь, регэксп не должен его ловить
_NON_SEXUAL_RE = re.compile(r"\bnon-?sexual\b", re.I)
# понятия из 18+ блока словаря, которые регэксп не распознаёт
NSFW_VOCAB_EXTRA = {"a bare chest", "a bare bottom"}


def is_nsfw_text(s):
    return bool(_NSFW_RE.search(_NON_SEXUAL_RE.sub("", s or "")))


def is_nsfw_vocab(en):
    return is_nsfw_text(en) or en.lower() in NSFW_VOCAB_EXTRA


PROMPT_TEMPLATES = ("a photo of {}", "a picture of {}", "an image of {}", "{}")

# ==========================================
# СЛОВАРЬ ТЕГОВ:  english | русский
# ==========================================
DEFAULT_VOCAB = """
# --- 18+ / NSFW (для детекции) ---
nude body | обнажённое тело
a nude photograph | обнажённое фото
a naked body | голое тело
explicit adult content | откровенный контент для взрослых
a sex scene | сексуальная сцена
genitals | гениталии
a bare chest | обнажённая грудь
a bare bottom | обнажённая задница
pornography | порнография
sexual intercourse | половой акт
a sexual act | сексуальный акт
a fetish photo | фетиш-фото
an erotic photo | эротическое фото
nudity | нагота
a nude person | обнажённый человек
a nude woman | обнажённая женщина
a nude man | обнажённый мужчина
sex | секс
oral sex | оральный секс
anal sex | анальный секс
penetration | проникновение
masturbation | мастурбация
cumshot | камшот
cum | сперма
ahegao | ахегао
lingerie | нижнее бельё
topless | топлес
bdsm | бдсм
bondage | связывание
erotic photography | эротика
hyper breasts | гипер грудь
huge breasts | огромная грудь
large breasts | большая грудь
hyper penis | гипер член
huge penis | огромный член
large testicles | большие яйца
small penis | маленький член
a penis | пенис
testicles | яички
a vagina | влагалище
an anus | анус
an ass | задница

# --- SFW-якоря (против ложных срабатываний) ---
a fully clothed person | полностью одетый человек
a person wearing clothes | человек в одежде
a modest person | скромный человек
safe for work content | безопасный контент
safe for work | безопасно для работы
family-friendly | семейный формат
wholesome | милый/добрый
fully clothed | полностью одет
modest clothing | скромная одежда
non-sexual | несексуальный
cute | милый
beautiful | красивый
handsome | привлекательный
elegant | элегантный
cozy | уютный
friendly | дружелюбный
professional | профессиональный
school appropriate | подходит для школы

# --- ЛЮДИ И ОДЕЖДА ---
a person | человек
a man | мужчина
a woman | женщина
a boy | мальчик
a girl | девочка
a teenager | подросток
an adult | взрослый
an elderly person | пожилой человек
a baby | младенец
a couple | пара
a group of people | группа людей
a crowd | толпа
a shirt | рубашка
a t-shirt | футболка
a hoodie | худи
a sweater | свитер
a jacket | куртка
a coat | пальто
a dress | платье
a skirt | юбка
pants | брюки
jeans | джинсы
shorts | шорты
socks | носки
shoes | обувь
sneakers | кроссовки
boots | ботинки
a hat | шляпа
a cap | кепка
a scarf | шарф
gloves | перчатки
glasses | очки
sunglasses | солнцезащитные очки
a watch | часы

# --- СЕМЬЯ И СОБЫТИЯ ---
a family | семья
friends | друзья
a child | ребёнок
parents | родители
a wedding | свадьба
a party | вечеринка
a holiday | праздник
a group photo | групповое фото
a portrait | портрет
a selfie | селфи
a birthday | день рождения
a celebration | празднование
a picnic | пикник
a concert | концерт
a festival | фестиваль
a graduation | выпускной
a meeting | встреча

# --- ЕДА, ЖИВОТНЫЕ, ПРИРОДА ---
food | еда
a cup of coffee | чашка кофе
tea | чай
water | вода
juice | сок
milk | молоко
bread | хлеб
cheese | сыр
fruit | фрукты
an apple | яблоко
a banana | банан
an orange | апельсин
berries | ягоды
vegetables | овощи
a salad | салат
soup | суп
pasta | паста
rice | рис
pizza | пицца
a sandwich | сэндвич
cake | торт
cookies | печенье
ice cream | мороженое
breakfast | завтрак
lunch | обед
dinner | ужин
a cat | кот
a dog | собака
a bird | птица
a fish | рыба
a rabbit | кролик
a hamster | хомяк
a horse | лошадь
a cow | корова
a sheep | овца
a goat | коза
a duck | утка
a chicken | курица
a butterfly | бабочка
a bee | пчела
a plant | растение
a flower | цветок
a tree | дерево
a forest | лес
mountains | горы
the sea | море
a beach | пляж
a river | река
a lake | озеро
a waterfall | водопад
the sky | небо
clouds | облака
the sun | солнце
the moon | луна
stars | звёзды
snow | снег
rain | дождь
a rainbow | радуга
a desert | пустыня
an island | остров
a cave | пещера
rocks | камни
grass | трава
leaves | листья
mushrooms | грибы

# --- СКРИНШОТЫ И ИНТЕРФЕЙСЫ ---
a screenshot | скриншот
a video game interface | игровой интерфейс
a menu screen | экран меню
text on screen | текст на экране
a comic page | страница комикса
a caption | подпись
a website | сайт
an app interface | интерфейс приложения
a chat window | окно чата
a browser | браузер
a desktop | рабочий стол
an icon | иконка
a button | кнопка
a window | окно
a dialog box | диалоговое окно
a graph | график
a chart | диаграмма
a presentation | презентация
a code editor | редактор кода
a terminal | терминал
a spreadsheet | таблица
a document page | страница документа
PDF | PDF
email | электронная почта
a notification | уведомление
a settings menu | меню настроек
a game HUD | игровой интерфейс
an inventory screen | экран инвентаря
a character selection | выбор персонажа
a level map | карта уровня
a scoreboard | табло

# --- СТИЛИ РИСОВАНИЯ ---
anime style | аниме стиль
manga style | манга стиль
3d render | 3д рендер
realistic | реалистично
cartoon style | мультяшный стиль
digital art | цифровое искусство
watercolor | акварель
oil painting | масляная живопись
acrylic painting | акриловая живопись
pencil sketch | карандашный набросок
ink drawing | тушь
line art | линейный рисунок
pixel art | пиксель-арт
low poly | лоу-поли
voxel art | воксель-арт
cel shading | сел-шейдинг
flat design | плоский дизайн
vector art | векторная графика
concept art | концепт-арт
illustration | иллюстрация
comic style | комиксный стиль
chibi | чиби
"""


def load_vocab(path=None):
    text = Path(path).read_text(encoding="utf-8") if path else DEFAULT_VOCAB
    seen, out = set(), []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        en, _, ru = line.partition("|")
        en, ru = en.strip(), ru.strip()
        key = en.lower()
        if en and key not in seen:
            seen.add(key)
            out.append((en, ru or en))
    return out


def map_dir(data, name=None):
    """Папка карты: корень data (полная карта) или data/maps/<name> (карта за период)."""
    data = Path(data)
    if not name:
        return data
    safe = re.sub(r"[^\w.\-]+", "_", name).strip("_")
    return data / "maps" / safe
