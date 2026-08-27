"""Religion-detection patterns that are not blind to the alphabet.

THE BUG THIS FIXES: the first versions of these patterns matched only each
language's native script. But people code-switch and transliterate constantly.
A Hindi speaker writes "bhagwan" in Latin letters, a Filipino writes "Diyos",
an Indonesian writes "Alhamdulillah", and half the world drops the English word
"God" into a sentence in its own language. Matching Devanagari alone put Hindi at
8% religious when a careful reader had found 41%. That is a four-fold error
produced entirely by an alphabet assumption, and it is the leading suspect for
Poland, the Philippines and Thailand reading implausibly secular too.

Every pattern below therefore has three layers:
  1. the native script
  2. the common romanisations of the same words
  3. the English religious vocabulary that shows up inside every language
"""
from __future__ import annotations
import re

# Layer 3: shows up inside comments in every language on earth.
SHARED = (
    r"\bgod\b|\bgods\b|jesus|christ|\bjesu|allah|\bpray|prayer|\bfaith\b|heaven|"
    r"\bsoul\b|church|bible|quran|koran|\bamen\b|bless|holy|divine|spiritual|"
    r"eternal|almighty|\blord\b|worship|\bsin\b|\bsoul|hallelujah|inshallah|"
    r"insha.?allah|alhamdulillah|mashallah|subhanallah|\bdua\b|\bimaan\b|\biman\b"
)

_NATIVE = {
 "ar": r"الله|ربنا|الرب|الجنة|النار|الصلا|الدعاء|القرآن|الاخرة|الآخرة|إيمان|ايمان|"
       r"النبي|\bرب\b|روح|توبة|استغفر|سبحان|الحمد|إنشاء الله|قدر|رزق|شهيد|دين",
 "fa": r"خدا|خداوند|الله|نماز|\bدعا|بهشت|\bروح\b|ایمان|قرآن|پیامبر|\bدین\b|معنوی|"
       r"توبه|آخرت|بهشتی|الهی|عبادت",
 "hi": r"भगवान|ईश्वर|प्रभु|परमात्मा|प्रार्थना|स्वर्ग|आत्मा|धर्म|पूजा|मोक्ष|गुरु|अल्लाह|"
       r"राम|कृष्ण|शिव|हरि|भक्ति|मंदिर|खुदा|नमाज|दुआ|पाप|पुण्य|कर्म",
 "th": r"พระเจ้า|พระพุทธ|พระธรรม|พระองค์|สวรรค์|วิญญาณ|ศรัทธา|บุญ|นิพพาน|ภาวนา|เทพ|"
       r"ทำบุญ|กรรม|สาธุ|อธิษฐาน|พระ",
 "vi": r"chúa|thượng đế|đức phật|\bphật\b|cầu nguyện|thiên đường|linh hồn|niết bàn|"
       r"đức tin|tâm linh|\bthần\b|\btrời\b|nghiệp|phúc|thánh|đạo",
 "ko": r"하나님|하느님|하늘|예수|기도|천국|신앙|믿음|교회|성경|부처|불교|극락|영혼|주님|"
       r"성령|은혜|축복|천당|아멘",
 "ja": r"神様|神さま|\b神\b|神を|神の|仏|仏様|祈り|お祈り|天国|信仰|お寺|極楽|魂|ご先祖|"
       r"因果|摂理|供養|成仏|南無",
 "ru": r"\bбог|бож|госпо|молитв|\bвера\b|верую|веру\b|душа|души|небеса|церк|\bрай\b|"
       r"христ|аллах|свят|грех|икон|молюсь|благослов",
 "zh": r"上帝|神明|\b神\b|佛|菩薩|祈禱|天堂|靈魂|信仰|教會|聖經|因果|業|修行",
}

_ROMAN = {
 "hi": r"bhagwan|bhagvan|ishwar|iishwar|parmatma|paramatma|prabhu|\bram\b|\brama\b|"
       r"krishna|krishn|\bshiv\b|shiva|\bhari\b|bhakti|mandir|khuda|namaz|prarthana|"
       r"\bpuja\b|\bpooja\b|moksh|dharma|\bkarma\b|jai shree|jai shri|\bsai\b|\batma\b|"
       r"\bom\b|guruji|waheguru|satnam",
 "ar": r"rabbi|rabbena|jannah|akhirah|akhira|salah|salat|tawba|astaghfir|subhan|"
       r"hamdulillah|qadar|\bdeen\b|shaheed",
 "id": r"allah|tuhan|\bdoa\b|surga|iman|ibadah|akhirat|sholat|shalat|agama|rohani|"
       r"berkah|taubat|tobat|syukur|yesus|kristus|gereja|\broh\b|\bdosa\b|malaikat|"
       r"insya|astaghfirullah|masya|nabi",
 "tr": r"allah|tanrı|tanri|\bdua\b|cennet|\biman|ibadet|ahiret|namaz|\bdin\b|\bruh\b|"
       r"peygamber|rabbim|rabbım|elhamdülillah|maşallah|inşallah|kader|tevekkül|"
       r"günah|kutsal|ilahi|amin",
 "fa": r"khoda|khuda|khodavand|namaz|behesht|rooh|payambar|allah|imaan|akherat|"
       r"tobeh|elahi",
 "th": r"phra chao|phrachao|phra phuttha|tham boon|\bboon\b|nipphan|nirvana|sattha|"
       r"\bkarma\b|\bsatu\b|athitan",
 "vi": r"\bchua\b|thuong de|\bphat\b|duc phat|cau nguyen|thien duong|linh hon|"
       r"niet ban|duc tin|tam linh|\bthan\b|\btroi\b|thanh|\bdao\b",
 "ko": r"hananim|haneunim|\byesu\b|kido|cheonguk|shinang|bucheo|\bamen\b",
 "ja": r"\bkami\b|kamisama|hotoke|\binori\b|tengoku|shinko|namu|gokuraku",
 "ru": r"\bbog\b|bozhe|gospodi|gospod|molitva|vera|dusha|tserkov|khristos|allah|"
       r"svyat|grekh|blagoslov",
 "de": r"\bgott|gottes|jesus|christ|glaub|gebet|beten|\bbete\b|himmel|seele|kirche|"
       r"bibel|göttlich|goettlich|\bewig|heilig|\bherr\b|sünde|suende|segen|allah",
 "fr": r"\bdieu|jésus|jesus|christ|\bfoi\b|prière|priere|prier|\bciel\b|\bâme\b|\bame\b|"
       r"église|eglise|bible|divin|éternel|eternel|sacré|sacre|seigneur|péché|"
       r"beni|allah",
 "es": r"\bdios|jesús|jesus|cristo|\bfe\b|oración|oracion|orar|\bcielo\b|\balma\b|"
       r"iglesia|biblia|divin|etern|sagrad|señor|senor|pecado|bendi|espírit|espirit|"
       r"virgen|santo|santa|rezar|allah",
 "pt": r"\bdeus|jesus|cristo|\bfé\b|\bfe\b|oração|oracao|orar|\bcéu\b|\bceu\b|\balma\b|"
       r"igreja|bíblia|biblia|divin|etern|sagrad|senhor|pecado|benç|benc|espírit|"
       r"espirit|santo|santa|rezar|aleluia|allah",
 "pl": r"\bbóg\b|\bbog\b|boga|bogu|bogiem|jezus|chrystus|wiar|modli|modlit|niebo|"
       r"dusza|duszy|kości[oó]ł|kosciol|bibli|świę|swie|wieczn|duchow|matka boska|"
       r"grzech|błogos|blogos|amen",
 "tl": r"diyos|dios|panginoon|jesus|hesus|kristo|pananampalataya|panalangin|langit|"
       r"kaluluwa|simbahan|banal|espirit|\bdasal\b|bathala|\bamen\b|biyaya|"
       r"salamat sa diyos|kasalanan|allah",
 "en": r"",  # SHARED already is the English layer
 "id_x": r"",
}


def pattern(lang: str) -> re.Pattern:
    """Native script, romanisation, and code-switched English, all at once."""
    parts = [SHARED]
    if lang in _NATIVE:
        parts.append(_NATIVE[lang])
    if lang in _ROMAN and _ROMAN[lang]:
        parts.append(_ROMAN[lang])
    return re.compile("|".join(p for p in parts if p), re.I)


LANGS = ["ar", "fa", "id", "es", "ko", "pt", "en", "hi", "tr", "fr", "ru", "de",
         "pl", "tl", "th", "ja", "vi"]
