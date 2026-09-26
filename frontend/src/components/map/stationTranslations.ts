/**
 * English renderings of the Hebrew STATION data (names, addresses, types)
 * seeded from the government fire-station registry, so the UI reads in one
 * language. Presentation only - the stored values are never modified.
 *
 * Scope: this static dictionary covers the fixed, known vocabulary in the
 * seeded station dataset, and is the primary translator for that data.
 * Dynamic, freeform location names (wildfire event/news/satellite-hotspot
 * location names) come back from the backend already in English, translated
 * once at ingestion via an LLM (see NewsMonitoringAgent.translate_report /
 * SimulationEventExecutor's TextProcessor wiring) - this dictionary is never
 * used to re-translate or second-guess a correct backend translation.
 *
 * `translateIfUntranslated` is the one narrow exception: a safety net for
 * when that backend translation step permanently failed (news_client.py's
 * translate_report/translate_location_name are deliberately best-effort and
 * never block a save - on a genuine, retry-exhausted LLM failure they
 * persist the original Hebrew text rather than lose the report, logging
 * "TRANSLATION FALLBACK TRIGGERED"). It only ever touches text that still
 * contains Hebrew characters, so correctly-translated text is never
 * re-processed - see its own docstring below.
 *
 * Translation is dictionary-based and word-by-word: every word in the seed
 * data (backend/src/database/data/firefighting_stations.json) has an entry,
 * so those stations translate fully. A word that is NOT in the dictionary
 * (e.g. a station added later) is romanized letter-by-letter as a rough,
 * always-Latin fallback - readable enough to identify, but it should get a
 * proper entry here. `untranslatedWords` lists such words.
 */

const HEBREW_WORD = /[֐-׿][֐-׿"'״׳]*/g;

const WORDS: Record<string, string> = {
  // --- descriptors ---
  רחוב: "St.", שדרות: "Sderot", שד: "Sderot", דרך: "Derech", כביש: "Highway", צומת: "Junction", מחלף: "Interchange",
  כיכר: "Sq.", נחל: "Nahal", מחנה: "Camp", מתחם: "Compound", חניון: "Parking", מושב: "Moshav", כפר: "Kfar",
  מועצה: "Council", אזורית: "Regional", האזורית: "Regional", פינת: "Corner of", סמוך: "Near", נמל: "Port", תעופה: "Airport",
  פארק: "Park", תעשיה: "Industrial", התעשייה: "Industry", "אזה\"ת": "Industrial Zone", ביטחון: "Security",
  מפרץ: "Bay", // e.g. Haifa Bay
  עליון: "Upper", תחתון: "Lower", מישור: "Plain", יער: "Forest",
  // --- major geographic regions (kept for station addresses that name a
  // wider region, e.g. a regional council address - not used for dynamic
  // wildfire event/news location names, which are backend-translated) ---
  הגולן: "HaGolan", הנגב: "HaNegev", ערבה: "Arava", הערבה: "HaArava",
  כנרת: "Kinneret", הכנרת: "HaKinneret", שפלה: "Shfela", השפלה: "HaShfela", ערה: "Ara",
  דן: "Dan", דרום: "South", צפון: "North", מרכז: "Center", מטה: "Mateh", החוף: "HaChof",
  "יו\"ש": "Judea and Samaria",
  // --- places & names (seed station names + address words) ---
  אגוז: "Agoz", ואדי: "Wadi", גוז: "Guz", אום: "Umm", אל: "Al", פחם: "Fahm", אופקים: "Ofakim", אור: "Or", יהודה: "Yehuda",
  סביון: "Savyon", עקיבא: "Akiva", איילון: "Ayalon", אילת: "Eilat", אלון: "Alon", אלעד: "Elad", אלקנה: "Elkana",
  אשדוד: "Ashdod", אשכול: "Eshkol", אשקלון: "Ashkelon", באקה: "Baqa", באקעה: "Baqa", גרביה: "Gharbiyye", באר: "Beer",
  יעקב: "Yaakov", שבע: "Sheva", מערב: "West", בורסה: "Bursa", בית: "Beit", חנינא: "Hanina", עטרות: "Atarot", שאן: "Shean",
  שמש: "Shemesh", בני: "Bnei", ברק: "Brak", בנימין: "Binyamin", בקעת: "Bikat", הירדן: "HaYarden", בת: "Bat", ים: "Yam",
  גוליס: "Julis", גב: "Gav", ההר: "HaHar", גבעת: "Givat", אולגה: "Olga", זאב: "Zeev", גבעתיים: "Givatayim", גדרה: "Gedera",
  גוש: "Gush", עציון: "Etzion", גליל: "Galil", גולן: "Golan", מרכזי: "Central", גן: "Gan", יבנה: "Yavne", דימונה: "Dimona",
  נגב: "Negev", מזרחי: "Eastern", מערבי: "Western", האומה: "HaUma", האלה: "HaEla", הבירה: "HaBira", הרים: "Harim",
  הרצליה: "Herzliya", השרון: "HaSharon", זבולון: "Zevulun", זכרון: "Zichron", חדרה: "Hadera", חולון: "Holon", חולות: "Holot",
  חורה: "Hura", חיפה: "Haifa", חצור: "Hatzor", הגלילית: "HaGlilit", חריש: "Harish", טבריה: "Tiberias", טירת: "Tirat",
  הכרמל: "HaCarmel", טמרה: "Tamra", יטבתה: "Yotvata", יכון: "Yachin", יפו: "Jaffa", יקנעם: "Yokneam", ירוחם: "Yeruham",
  כדורי: "Kaduri", כרמל: "Carmel", מגדל: "Migdal", העמק: "HaEmek", מודיעין: "Modiin", מירב: "Meirav", חוף: "Hof", מסעדה: "Mas'ade",
  מעוז: "Maoz", מעלות: "Maalot", מצודה: "Metzuda", מצפה: "Mitzpe", רמון: "Ramon", משואה: "Mashu'a", אורה: "Ora", מתתיהו: "Matityahu",
  נאות: "Neot", חובב: "Hovav", רמת: "Ramat", נהריה: "Nahariya", נוף: "Nof", הגליל: "HaGalil", נחשון: "Nachshon", נצרת: "Nazareth",
  נשר: "Nesher", נתיבות: "Netivot", נתניה: "Netanya", ספיר: "Sapir", עד: "Ad", הלום: "Halom", עוספיא: "Isfiya", עספיא: "Isfiya",
  עפולה: "Afula", יזרעאל: "Jezreel", עפרה: "Ofra", עראבה: "Arraba", ערד: "Arad", עתידים: "Atidim", פסגה: "Pisga", פתח: "Petah",
  תקווה: "Tikva", צור: "Tzur", באהר: "Baher", יגאל: "Yigal", ציון: "Zion", ב: "B", ציפורית: "Tzipori", צלמון: "Tzalmon",
  צמח: "Tzemach", צפת: "Safed", קדימה: "Kadima", קצרין: "Katzrin", קריות: "Krayot", קרית: "Kiryat", קריית: "Kiryat", ארבע: "Arba",
  גת: "Gat", הלאום: "HaLeom", טבעון: "Tivon", מלאכי: "Malakhi", קרני: "Karnei", שומרון: "Shomron", ראש: "Rosh", העין: "HaAyin",
  ראשון: "Rishon", לציון: "LeZion", רהט: "Rahat", רחובות: "Rehovot", רמה: "Rama", גן_: "Gan", שדרות_: "Sderot", שוהם: "Shoham",
  שחמון: "Shachmon", שפרעם: "Shefa-Amr", שקד: "Shaked", שרונה: "Sarona", תל: "Tel", אביב: "Aviv", תמנע: "Timna", תמר: "Tamar",
  תרדיון: "Tardiyon", משגב: "Misgav", ירושלים: "Jerusalem", קרית_: "Kiryat", אונו: "Ono", לוד: "Lod", עכו: "Acre",
  // --- street / personal names in the seed addresses ---
  יקות: "Yakut", חמווי: "Hamawi", מדינה: "Madina", הנשיא: "HaNasi", הכבאים: "HaKabaim", האילן: "HaIlan", "הרמ\"א": "HaRama",
  התמרים: "HaTmarim", המלך: "HaMelech", שלמה: "Shlomo", לוחמי: "Lohamei", האש: "HaEsh", מגן: "Magen", התחיה: "HaTechiya",
  ביר: "Bir", שמוליק: "Shmulik", בורג: "Burg", סעדיה: "Saadia", מלל: "Malal", אילן: "Ilan", הרקון: "HaYarkon", עמיחי: "Amichai",
  גידי: "Gidi", פייגלין: "Feiglin", יצחק: "Yitzhak", השבעה: "HaShiva", אהרונוביץ: "Aharonovitz", תבור: "Tavor", הצור: "HaTzur",
  מעלה: "Maale", אדומים: "Adumim", שלומציון: "Shlomtzion", אהוד: "Ehud", קינמון: "Kinamon", שייח: "Sheikh", אמון: "Amun",
  טריף: "Tarif", חורון: "Horon", הדייגים: "HaDayagim", התאנה: "HaTe'ena", המעיין: "HaMaayan", בן: "Ben", גוריון: "Gurion",
  ברזילי: "Barzilai", שמונה: "Shmona", החרושת: "HaCharoshet", כרמיאל: "Karmiel", הרצל: "Herzl", המלאכה: "HaMelacha",
  שמגר: "Shamgar", "השח\"ל": "HaShachal", מרדכי: "Mordechai", בר: "Bar", גיורא: "Giora", שבעת: "Shivat", הכוכבים: "HaKochavim",
  שאול: "Shaul", טשרניחובסקי: "Tchernichovsky", סבא: "Saba", דוד: "David", רמז: "Remez", אהרון: "Aharon", "צה\"ל": "IDF",
  הרוקמים: "HaRokmim", "אל-אסלאם": "Al-Islam", חלץ: "Halatz", המסגר: "HaMasger", אורן: "Oren", "לח\"י": "Lehi", פאטמה: "Fatima",
  "אל-זהראא": "Al-Zahra", דואני: "Duani", אוהל: "Ohel", שרה: "Sarah", ביתר: "Beitar", עילית: "Illit", יד: "Yad", חנה: "Hanna",
  אומץ: "Omez", צבי: "Tzvi", האיריסים: "HaIrisim", "אלכחייל": "Al-Kahil", משה: "Moshe", היינריך: "Heinrich", היינה: "Heine",
  הצאלון: "HaTzalon", עמק: "Emek", מכבים: "Maccabim", רעות: "Reut", היצירה: "HaYetzira", מבשרת: "Mevaseret", תרשיחא: "Tarshiha",
  אנילביץ: "Anielewicz", החרש: "HaCharash", ברעם: "Bar'am", חווה: "Hava", המרפא: "HaMarpe", הר: "Har", חוצבים: "Hotzvim",
  חשמונאים: "HaHashmonaim", הדקל: "HaDekel", יחיעם: "Yehiam", דליה: "Dalia", א: "A", הזית: "HaZayit", צלפון: "Tzalafon",
  שוהדא: "Shuhada", השלום: "HaShalom", אחת: "Achat", עשרה: "Esre", הנקודות: "HaNekudot", הרכבת: "HaRakevet", הרפואה: "HaRefua",
  אבא: "Abba", חושי: "Hushi", עילוט: "Ilut", החטיבה: "HaHativa", הברזל: "HaBarzel", המכבים: "HaMaccabim", "אלמדינה": "Al-Madina",
  "אלמונאווורה": "Al-Munawwara", גרניט: "Granit", שמוטקין: "Shmotkin", הגדוד: "HaGdud", השלישי: "HaShlishi", צורן: "Tzoran",
  השוקת: "HaShoket", "ח\"ן": "Chen", ביאליק: "Bialik", גור: "Gur", אריה: "Arye", לכיש: "Lachish", רבין: "Rabin", אלעזר: "Elazar",
  הרב: "Rabbi", חיים: "Haim", פינטו: "Pinto", רחבעם: "Rehavam", זאבי: "Zeevi", אחוב: "Ahuv", עומר: "Omer", מוכתר: "Mukhtar",
  הים: "HaYam", דולב: "Dolev", אליעזר: "Eliezer", הלל: "Hillel", אריאל: "Ariel", שרון: "Sharon", החבל: "HaHevel",
  הנחשונים: "HaNachshonim", ששת: "Sheshet", הימים: "HaYamim", סולטן: "Sultan", באשא: "Pasha", אטראש: "Atrash", מופק: "Mufaq",
  דיאב: "Diab", הקידה: "HaKida", טל: "Tal", מנשה: "Menashe", ליאונרדו: "Leonardo", דה: "da", וינצי: "Vinci", אבן: "Ibn",
  אסלאם: "Islam", זהראא: "Zahra", יוקנעם: "Yokneam", תקוה: "Tikva", גבירול: "Gvirol", ורדימון: "Vardimon", תכלת: "Tekhelet", לניצנה: "Nitzana",
};

// Multi-word phrases translated as a unit (checked before word-by-word).
const PHRASES: [RegExp, string][] = [
  [/למועצה האזורית/g, "Regional Council"],
  [/מועצה אזורית/g, "Regional Council"],
  [/נמל תעופה/g, "Airport"],
  // Hebrew puts the qualifier first; English reads it the other way round.
  [/מפרץ חיפה/g, "Haifa Bay"],
  [/פארק הכרמל/g, "Carmel Park"],
  [/יער ירושלים/g, "Jerusalem Forest"],
  [/גליל מערבי/g, "Western Galilee"],
  [/גליל מזרחי/g, "Eastern Galilee"],
  [/הגליל העליון/g, "Upper Galilee"],
  [/הגליל התחתון/g, "Lower Galilee"],
  [/גליל עליון/g, "Upper Galilee"],
  [/גליל תחתון/g, "Lower Galilee"],
  // A well-known named region: the internationally recognized English name
  // reads better for an operator than a mechanical "Ramat HaGolan" would.
  [/רמת הגולן/g, "Golan Heights"],
  // "ו" ("and") prefixes "שומרון" here, so the two words don't tokenize
  // separately - handled as one phrase rather than trying to strip
  // conjunction prefixes generically.
  [/יהודה ושומרון/g, "Judea and Samaria"],
  [/נציבות כבאות והצלה לישראל/g, "Israel Fire and Rescue Commission"],
];

const STATION_TYPES: Record<string, string> = {
  אזורית: "Regional",
  משנה: "Sub-station",
  מחוז: "District",
  מטה: "Headquarters",
};

const ROMAN: Record<string, string> = {
  א: "a", ב: "b", ג: "g", ד: "d", ה: "h", ו: "v", ז: "z", ח: "ch", ט: "t", י: "y", כ: "k", ך: "kh", ל: "l", מ: "m", ם: "m",
  נ: "n", ן: "n", ס: "s", ע: "a", פ: "p", ף: "f", צ: "tz", ץ: "tz", ק: "k", ר: "r", ש: "sh", ת: "t",
};

function romanize(word: string): string {
  const latin = Array.from(word)
    .map((ch) => ROMAN[ch] ?? (/[֐-׿]/.test(ch) ? "" : ch))
    .join("");
  return latin.charAt(0).toUpperCase() + latin.slice(1);
}

function translateWord(word: string): string {
  return WORDS[word] ?? romanize(word);
}

function translateFragment(text: string): string {
  let out = text;
  for (const [pattern, english] of PHRASES) {
    out = out.replace(pattern, english);
  }
  return out.replace(HEBREW_WORD, translateWord).replace(/\s+/g, " ").trim();
}

/** Hebrew words in `text` that have no dictionary entry (they would be romanized). */
export function untranslatedWords(text: string): string[] {
  let scrubbed = text;
  for (const [pattern] of PHRASES) {
    scrubbed = scrubbed.replace(pattern, " ");
  }
  return (scrubbed.match(HEBREW_WORD) ?? []).filter((word) => !(word in WORDS));
}

/** English name of a station without the "Station" suffix (for lists: "Nesher"). */
export function translateStationLabel(name: string): string {
  return translateFragment(name);
}

export function translateStationName(name: string): string {
  const english = translateFragment(name);
  return /station/i.test(english) ? english : `${english} Station`;
}

/**
 * Safety net for dynamic (non-station) text the backend was supposed to
 * translate but didn't - see the module docstring above. Text that is
 * already in English (the normal case, always) passes through completely
 * unchanged. Text that still contains Hebrew is translated only when the
 * dictionary covers EVERY Hebrew word (e.g. a known place name such as
 * "יער ירושלים"); otherwise the original Hebrew is returned as-is. Free
 * text (news headlines/summaries) is never romanized letter-by-letter -
 * readable Hebrew beats unreadable transliteration ("Dyvchym Rashvnym...").
 * Never re-translates or alters text the backend already translated correctly.
 */
export function translateIfUntranslated(text: string): string;
export function translateIfUntranslated(text: string | null): string | null;
export function translateIfUntranslated(text: string | null): string | null {
  if (text === null || !/[֐-׿]/.test(text)) {
    return text;
  }
  const placeName = SIMULATION_PLACE_NAMES[text.trim()];
  if (placeName !== undefined) {
    return placeName;
  }
  return untranslatedWords(text).length === 0 ? translateFragment(text) : text;
}

/**
 * English names of the demo areas, keyed by the exact Hebrew `location_name`
 * the simulator's news reports carry (backend news_data_generator
 * _LOCATION_REPORT_NAMES; a backend test keeps both lists in sync). Matched
 * only as the WHOLE string, so station vocabulary such as "נוף הגליל"
 * ("Nof HaGalil") or "טירת הכרמל" is unaffected.
 */
export const SIMULATION_PLACE_NAMES: Readonly<Record<string, string>> = {
  הכרמל: "Carmel",
  "יער ירושלים": "Jerusalem Forest",
  הגליל: "Galilee",
  "רמת הגולן": "Golan Heights",
  "הרי יהודה": "Judean Hills",
};
/** House-number suffix letters become Latin (13א -> 13a). */
function houseNumber(raw: string): string {
  return raw.replace(/[א-ת]/g, (letter) => ROMAN[letter] ?? "");
}


/**
 * Hand-written English for addresses that don't follow "Street 12 City"
 * (intersections, industrial zones, highways, moshav/council addresses).
 * Keys are the Hebrew address with whitespace collapsed; these take
 * precedence over the generic splitting below and are written City-first.
 */
const ADDRESS_OVERRIDES: Record<string, string> = {
  "אל מדינה, אום אל פחם": "Umm al-Fahm, Al-Madina",
  "ביר באקה, באקעה אל גרביה": "Baqa al-Gharbiyye, Bir Baqa",
  "שלומציון בקעת הירדן": "Bikat HaYarden, Shlomtzion",
  "מחנה חורון, מתחם ביטחון גב ההר": "Gav HaHar, Horon Camp Security Compound",
  "כביש 367, צומת גוש עציון": "Gush Etzion, Highway 367 Junction",
  "כביש 38 צומת האלה": "HaEla Junction, Highway 38",
  "כביש 211, סמוך לניצנה": "Nitzana, Highway 211 (near Nitzana)",
  "אל-אסלאם חורה": "Hura, Al-Islam",
  "רחוב אורן חריש": "Harish, Oren St.",
  "פאטמה אל-זהראא, טמרה": "Tamra, Fatima Al-Zahra",
  "חניון יטבתה": "Yotvata, Yotvata Parking",
  "יד חנה, מושב אומץ": "Moshav Omez, Yad Hanna",
  "דרך משה בר יהודה כדורי": "Kaduri, Derech Moshe Bar Yehuda",
  "עמק האלה, מודיעין מכבים רעות": "Modiin-Maccabim-Reut, Emek HaEla",
  "מושב מירב, מועצה אזורית חוף הכרמל": "Hof HaCarmel Regional Council, Moshav Meirav",
  "המרפא 19, הר חוצבים, ירושלים": "Jerusalem, Har Hotzvim, HaMarpe 19",
  "שד הדקל 1, מועצה אזורית נאות חובב": "Neot Hovav Regional Council, Sderot HaDekel 1",
  "גרניט 1 אזה\"ת צור יגאל": "Tzur Yigal (Industrial Zone), Granit 1",
  "התעשייה 33 ערד": "Arad, HaTaasiya 33",
  "התעשייה 1 נוף הגליל": "Nof HaGalil, HaTaasiya 1",
  "הקידה, טל מנשה": "Tal Menashe, HaKida",
  "הזית, מושב צלפון": "Moshav Tzalafon, HaZayit",
  "אחת עשרה הנקודות נתיבות": "Netivot, Achat Esre HaNekudot",
  "סמוך למועצה האזורית ספיר": "Sapir Regional Council (near)",
  "מסעדה, כביש 98": "Mas'ade, Highway 98",
  "כביש 446 כיכר חשמונאים": "Matityahu, Highway 446 / HaHashmonaim Sq.",
  "מחלף צלמון, כביש 807": "Tzalmon, Highway 807 / Tzalmon Interchange",
  "צומת צמח כביש 92": "Tzemach, Highway 92 / Tzemach Junction",
  "אחוב עומר אל מוכתר": "Rahat, Ahuv Omer Al-Mukhtar",
  "סולטן באשא אל אטראש פינת מופק דיאב שפרעם": "Shefa-Amr, Sultan Pasha Al Atrash / Mufaq Diab",
  "נמל תעופה רמון, תמנע": "Timna, Ramon Airport",
  "תכלת 1 פארק תעשיה משגב-תרדיון": "Misgav (Tardiyon), Tekhelet 1",
};

/**
 * Translates an address, reordering the common Hebrew "Street 12 City" shape
 * into English "City, Street 12". Anything else is translated word by
 * word and never leaves stray spaces or commas.
 */
export function translateStationAddress(address: string): string {
  const trimmed = address.trim().replace(/\s+/g, " ");
  const override = ADDRESS_OVERRIDES[trimmed];
  if (override) {
    return override;
  }
  // Comma-separated Hebrew addresses run "street, city": flip to city-first.
  if (trimmed.includes(",")) {
    return trimmed
      .split(",")
      .map((part) => translateAddressPart(part))
      .filter(Boolean)
      .reverse()
      .join(", ");
  }
  return translateAddressPart(trimmed);
}

function translateAddressPart(part: string): string {
  const trimmed = part.trim();
  const withCity = /^(.+?)\s+(\d+[א-ת]?)\s+(\S.*)$/.exec(trimmed);
  if (withCity) {
    return `${translateFragment(withCity[3])}, ${translateFragment(withCity[1])} ${houseNumber(withCity[2])}`;
  }
  const streetOnly = /^(.+?)\s+(\d+[א-ת]?)$/.exec(trimmed);
  if (streetOnly) {
    return `${translateFragment(streetOnly[1])} ${houseNumber(streetOnly[2])}`;
  }
  return translateFragment(trimmed);
}

export function translateStationType(type: string): string {
  return STATION_TYPES[type.trim()] ?? translateFragment(type);
}
