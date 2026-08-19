/**
 * PhytoShield AI — Centralized Human-Friendly Disease Label Normalizer
 * 
 * Maps raw ML model training identifiers (e.g. Pepper__bell___Bacterial_spot)
 * to human-readable, agronomically accurate, professional display names
 * (e.g. Bell Pepper Bacterial Spot Disease).
 */

export const CURATED_DISEASE_NAMES: Record<string, string> = {
  // ── Pepper (Capsicum annuum) ───────────────────────────────────────────────
  'Pepper__bell___Bacterial_spot': 'Bell Pepper Bacterial Spot Disease',
  'Pepper_bell___Bacterial_spot':  'Bell Pepper Bacterial Spot Disease',
  'Pepper__bell___healthy':         'Healthy Bell Pepper Foliage',
  'Pepper_bell___healthy':          'Healthy Bell Pepper Foliage',

  // ── Potato (Solanum tuberosum) ─────────────────────────────────────────────
  'Potato___Early_blight':          'Potato Early Blight Disease',
  'Potato___Late_blight':           'Potato Late Blight Disease',
  'Potato___healthy':               'Healthy Potato Foliage',
  'Potato_healthy':                 'Healthy Potato Foliage',

  // ── Tomato (Solanum lycopersicum) ──────────────────────────────────────────
  'Tomato_Bacterial_spot':                          'Tomato Bacterial Spot Disease',
  'Tomato___Bacterial_spot':                        'Tomato Bacterial Spot Disease',
  'Tomato_Early_blight':                            'Tomato Early Blight Disease',
  'Tomato___Early_blight':                          'Tomato Early Blight Disease',
  'Tomato_Late_blight':                             'Tomato Late Blight Disease',
  'Tomato___Late_blight':                           'Tomato Late Blight Disease',
  'Tomato_Leaf_Mold':                               'Tomato Leaf Mold Disease',
  'Tomato___Leaf_Mold':                             'Tomato Leaf Mold Disease',
  'Tomato_Septoria_leaf_spot':                      'Tomato Septoria Leaf Spot Disease',
  'Tomato___Septoria_leaf_spot':                    'Tomato Septoria Leaf Spot Disease',
  'Tomato_Spider_mites_Two_spotted_spider_mite':    'Tomato Two-Spotted Spider Mite Infestation',
  'Tomato___Spider_mites_Two_spotted_spider_mite':  'Tomato Two-Spotted Spider Mite Infestation',
  'Tomato__Target_Spot':                            'Tomato Target Spot Disease',
  'Tomato___Target_Spot':                           'Tomato Target Spot Disease',
  'Tomato__Tomato_YellowLeaf__Curl_Virus':          'Tomato Yellow Leaf Curl Virus (TYLCV)',
  'Tomato___Tomato_YellowLeaf__Curl_Virus':         'Tomato Yellow Leaf Curl Virus (TYLCV)',
  'Tomato_Yellow_Leaf_Curl_Virus':                  'Tomato Yellow Leaf Curl Virus (TYLCV)',
  'Tomato__Tomato_mosaic_virus':                    'Tomato Mosaic Virus (ToMV)',
  'Tomato___Tomato_mosaic_virus':                   'Tomato Mosaic Virus (ToMV)',
  'Tomato_mosaic_virus':                            'Tomato Mosaic Virus (ToMV)',
  'Tomato_healthy':                                 'Healthy Tomato Foliage',
  'Tomato___healthy':                               'Healthy Tomato Foliage',

  // ── Apple (Malus domestica) ────────────────────────────────────────────────
  'Apple___Apple_scab':             'Apple Scab Disease',
  'Apple___Black_rot':              'Apple Black Rot Disease',
  'Apple___Cedar_apple_rust':       'Cedar Apple Rust Disease',
  'Apple___healthy':                'Healthy Apple Foliage',

  // ── Corn (Zea mays) ────────────────────────────────────────────────────────
  'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': 'Corn Gray Leaf Spot Disease',
  'Corn_(maize)___Common_rust_':                        'Corn Common Rust Disease',
  'Corn_(maize)___Northern_Leaf_Blight':                'Northern Corn Leaf Blight Disease',
  'Corn_(maize)___healthy':                             'Healthy Corn Foliage',

  // ── Grape (Vitis vinifera) ─────────────────────────────────────────────────
  'Grape___Black_rot':                              'Grapevine Black Rot Disease',
  'Grape___Esca_(Black_Measles)':                   'Grapevine Esca (Black Measles) Disease',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':     'Grapevine Leaf Blight (Isariopsis Spot)',
  'Grape___healthy':                                'Healthy Grapevine Foliage',

  // ── Peach (Prunus persica) ─────────────────────────────────────────────────
  'Peach___Bacterial_spot':                         'Peach Bacterial Spot Disease',
  'Peach___healthy':                                'Healthy Peach Foliage',

  // ── Strawberry (Fragaria × ananassa) ───────────────────────────────────────
  'Strawberry___Leaf_scorch':                       'Strawberry Leaf Scorch Disease',
  'Strawberry___healthy':                           'Healthy Strawberry Foliage',

  // ── Cherry (Prunus serotina / avium) ───────────────────────────────────────
  'Cherry_(including_sour)___Powdery_mildew':       'Cherry Powdery Mildew Disease',
  'Cherry_(including_sour)___healthy':              'Healthy Cherry Foliage',
};

/**
 * Normalizes a raw model class identifier into a polished, human-friendly display name.
 * 
 * @param rawClassName The raw class name output from the ML classifier (e.g. 'Pepper__bell___Bacterial_spot')
 * @returns Human-friendly name (e.g. 'Bell Pepper Bacterial Spot Disease')
 */
export function formatDiseaseDisplayName(rawClassName: string | null | undefined): string {
  if (!rawClassName) return 'Unknown Plant Specimen';

  const trimmed = rawClassName.trim();

  // 1. Direct Curated Mapping Lookup
  if (CURATED_DISEASE_NAMES[trimmed]) {
    return CURATED_DISEASE_NAMES[trimmed];
  }

  // 2. Case-insensitive / Normalized Key Lookup
  const normalizedKey = trimmed.replace(/\s+/g, '_');
  for (const [key, val] of Object.entries(CURATED_DISEASE_NAMES)) {
    if (key.toLowerCase() === normalizedKey.toLowerCase()) {
      return val;
    }
  }

  // 3. Fallback Semantic Normalization
  let clean = trimmed
    .replace(/___+/g, ' - ')
    .replace(/__+/g, ' ')
    .replace(/_+/g, ' ')
    .replace(/\s*\(including\s+sour\)/gi, '')
    .replace(/\s*\(maize\)/gi, '')
    .trim();

  // Normalize specific crop phrases
  clean = clean.replace(/\bPepper\s+bell\b/gi, 'Bell Pepper');
  clean = clean.replace(/\bYellowLeaf\b/gi, 'Yellow Leaf');
  clean = clean.replace(/\bTwo\s+spotted\s+spider\s+mite\b/gi, 'Two-Spotted Spider Mite');

  // Handle healthy cases
  if (/healthy/i.test(clean)) {
    const crop = clean.replace(/healthy/i, '').replace(/[-:]/g, '').trim();
    return crop ? `Healthy ${crop} Foliage` : 'Healthy Plant Foliage';
  }

  // Ensure Title Case on fallback words
  clean = clean
    .split(/\s+/)
    .map(word => {
      if (/^(and|of|or|the|in|with)$/i.test(word)) return word.toLowerCase();
      if (/^(TYLCV|ToMV|XAI|AI|DNA|RNA|IG|CAM|SHAP|LIME)$/i.test(word)) return word.toUpperCase();
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');

  // Append Disease if it is an unclassified pathology and doesn't end with Disease / Virus / Mite
  if (!/(disease|virus|infestation|mildew|blight|spot|rot|scab|rust|mold|foliage)$/i.test(clean)) {
    clean += ' Disease';
  }

  return clean;
}

/**
 * Helper to check if a prediction represents healthy plant foliage.
 */
export function isHealthyCondition(rawClassName: string): boolean {
  return /healthy/i.test(rawClassName);
}
