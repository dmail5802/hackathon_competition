/*
  Backup ML model (Logistic Regression equation).

  Updated coefficients from your fit:
  intercept: -0.3665506210496277
  Thallium: 1.157993707086125
  Chest pain type: 1.034096959927882
  Exercise angina: 0.7143597238023591
  Cholesterol: 0.12131236470510319
  BP: -0.0022169972701850745
  Sex: 0.55792322715737

  logit(p) = b0
           + b_thallium * Thallium
           + b_chest    * ChestPain
           + b_angina   * ExerciseAngina
           + b_chol     * Cholesterol
           + b_bp       * BP
           + b_sex      * Sex

  p = sigmoid(logit)
*/

const LOGISTIC_COEFFS = {
  intercept: -0.3665506210496277,
  thallium: 1.157993707086125,
  chestPain: 1.034096959927882,
  exerciseAngina: 0.7143597238023591,
  cholesterol: 0.12131236470510319,
  bp: -0.0022169972701850745,
  sex: 0.55792322715737,
};

function sigmoid(z) {
  const clipped = Math.max(-60, Math.min(60, z));
  return 1 / (1 + Math.exp(-clipped));
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function riskBand(p) {
  if (p < 0.30) return "Low";
  if (p < 0.65) return "Moderate";
  return "High";
}

// DOM
const mode = document.getElementById("mode");

const thallium = document.getElementById("thallium");
const chestPain = document.getElementById("chestPain");
const threshold = document.getElementById("threshold");
const cholesterol = document.getElementById("cholesterol");
const bp = document.getElementById("bp");

const thalliumVal = document.getElementById("thalliumVal");
const chestPainVal = document.getElementById("chestPainVal");
const thresholdVal = document.getElementById("thresholdVal");
const cholesterolVal = document.getElementById("cholesterolVal");
const bpVal = document.getElementById("bpVal");

const anginaNo = document.getElementById("anginaNo");
const anginaYes = document.getElementById("anginaYes");
const sex0Btn = document.getElementById("sex0");
const sex1Btn = document.getElementById("sex1");

const probPct = document.getElementById("probPct");
const gaugeFill = document.getElementById("gaugeFill");
const resultBadge = document.getElementById("resultBadge");
const riskBandEl = document.getElementById("riskBand");
const modeBadge = document.getElementById("modeBadge");

const trainAiBtn = document.getElementById("trainAiBtn");
const resetAiBtn = document.getElementById("resetAiBtn");
const trainStatus = document.getElementById("trainStatus");

const equationText = document.getElementById("equationText");

// State
let exerciseAngina = 0;
let sex = 0;

// AI model state
let aiModel = null;
let aiNorm = null; // {min:[], max:[]} for min-max scaling

const FEATURES = ["Thallium", "Chest pain type", "Exercise angina", "Cholesterol", "BP", "Sex"];

function updateEquationText() {
  const b = LOGISTIC_COEFFS;
  equationText.textContent =
    `logit(p) = ${b.intercept.toFixed(4)}`
    + ` + ${b.thallium.toFixed(4)}·Thallium`
    + ` + ${b.chestPain.toFixed(4)}·ChestPain`
    + ` + ${b.exerciseAngina.toFixed(4)}·ExerciseAngina`
    + ` + ${b.cholesterol.toFixed(4)}·Cholesterol`
    + ` + ${b.bp.toFixed(4)}·BP`
    + ` + ${b.sex.toFixed(4)}·Sex`;
}

function updateToggleButtons() {
  // Exercise angina
  if (exerciseAngina === 0) {
    anginaNo.classList.add("active");
    anginaNo.setAttribute("aria-pressed", "true");
    anginaYes.classList.remove("active");
    anginaYes.setAttribute("aria-pressed", "false");
  } else {
    anginaYes.classList.add("active");
    anginaYes.setAttribute("aria-pressed", "true");
    anginaNo.classList.remove("active");
    anginaNo.setAttribute("aria-pressed", "false");
  }

  // Sex
  if (sex === 0) {
    sex0Btn.classList.add("active");
    sex0Btn.setAttribute("aria-pressed", "true");
    sex1Btn.classList.remove("active");
    sex1Btn.setAttribute("aria-pressed", "false");
  } else {
    sex1Btn.classList.add("active");
    sex1Btn.setAttribute("aria-pressed", "true");
    sex0Btn.classList.remove("active");
    sex0Btn.setAttribute("aria-pressed", "false");
  }
}

function readInputs() {
  return {
    Thallium: Number(thallium.value),
    "Chest pain type": Number(chestPain.value),
    "Exercise angina": Number(exerciseAngina),
    Cholesterol: Number(cholesterol.value),
    BP: Number(bp.value),
    Sex: Number(sex),
    threshold: Number(threshold.value),
  };
}

function logisticPredictProbability(x) {
  const b = LOGISTIC_COEFFS;
  const logit =
    b.intercept
    + b.thallium * x.Thallium
    + b.chestPain * x["Chest pain type"]
    + b.exerciseAngina * x["Exercise angina"]
    + b.cholesterol * x.Cholesterol
    + b.bp * x.BP
    + b.sex * x.Sex;

  return sigmoid(logit);
}

function minMaxScale(vec, minArr, maxArr) {
  return vec.map((v, i) => {
    const mn = minArr[i];
    const mx = maxArr[i];
    if (mx === mn) return 0;
    return (v - mn) / (mx - mn);
  });
}

function aiPredictProbability(x) {
  if (!aiModel || !aiNorm) return null;

  const raw = [
    x.Thallium,
    x["Chest pain type"],
    x["Exercise angina"],
    x.Cholesterol,
    x.BP,
    x.Sex
  ];

  const scaled = minMaxScale(raw, aiNorm.min, aiNorm.max);
  const inputTensor = tf.tensor2d([scaled], [1, scaled.length]);
  const out = aiModel.predict(inputTensor);
  const p = out.dataSync()[0];
  inputTensor.dispose();
  out.dispose();
  return p;
}

function render(p, thresholdValNow, usedMode) {
  const pClamped = clamp01(p);
  probPct.textContent = `${Math.round(pClamped * 100)}%`;
  gaugeFill.style.height = `${Math.round(pClamped * 100)}%`;

  const pred = pClamped >= thresholdValNow ? 1 : 0;
  const band = riskBand(pClamped);

  riskBandEl.textContent = `Risk band: ${band} (threshold = ${thresholdValNow.toFixed(2)})`;
  modeBadge.textContent = `Mode: ${usedMode}`;

  if (pred === 1) {
    resultBadge.textContent = "Predicted: Presence";
    resultBadge.style.borderColor = "rgba(255,77,90,.55)";
    resultBadge.style.background = "rgba(255,77,90,.12)";
  } else {
    resultBadge.textContent = "Predicted: Absence";
    resultBadge.style.borderColor = "rgba(31,191,117,.55)";
    resultBadge.style.background = "rgba(31,191,117,.12)";
  }
}

function predictAndUpdate() {
  const x = readInputs();

  thalliumVal.textContent = x.Thallium.toFixed(0);
  chestPainVal.textContent = x["Chest pain type"].toFixed(0);
  cholesterolVal.textContent = x.Cholesterol.toFixed(0);
  bpVal.textContent = x.BP.toFixed(0);
  thresholdVal.textContent = x.threshold.toFixed(2);

  const selectedMode = mode.value;

  if (selectedMode === "ai") {
    const pAi = aiPredictProbability(x);
    if (pAi === null) {
      const pFallback = logisticPredictProbability(x);
      render(pFallback, x.threshold, "AI (not trained → Backup Logistic)");
      return;
    }
    render(pAi, x.threshold, "AI (Neural Net)");
    return;
  }

  const p = logisticPredictProbability(x);
  render(p, x.threshold, "Backup Logistic");
}

/* ---------------------------
   AI TRAINING (browser)
   - Loads CSV from ../data/heart_disease/train.csv
   - Uses FEATURES: Thallium, Chest pain type, Exercise angina, Cholesterol, BP, Sex
   - Target: Heart Disease (Presence/Absence)
--------------------------- */

async function trainAiFromCsv() {
  trainStatus.textContent = "AI status: loading CSV...";
  const csvPath = "../data/heart_disease/train.csv";

  const resp = await fetch(csvPath);
  if (!resp.ok) {
    trainStatus.textContent = `AI status: failed to load CSV (${resp.status}). Check path: ${csvPath}`;
    return;
  }
  const csvText = await resp.text();

  trainStatus.textContent = "AI status: parsing CSV...";

  const parsed = Papa.parse(csvText, { header: true, dynamicTyping: true, skipEmptyLines: true });
  if (parsed.errors && parsed.errors.length) {
    trainStatus.textContent = `AI status: CSV parse error: ${parsed.errors[0].message}`;
    return;
  }

  const rows = parsed.data;

  const Xraw = [];
  const yraw = [];

  for (const r of rows) {
    const target = r["Heart Disease"];
    if (target !== "Presence" && target !== "Absence") continue;

    const featVec = [
      Number(r["Thallium"]),
      Number(r["Chest pain type"]),
      Number(r["Exercise angina"]),
      Number(r["Cholesterol"]),
      Number(r["BP"]),
      Number(r["Sex"]),
    ];

    if (featVec.some(v => Number.isNaN(v))) continue;

    Xraw.push(featVec);
    yraw.push(target === "Presence" ? 1 : 0);
  }

  if (Xraw.length < 100) {
    trainStatus.textContent = "AI status: not enough valid rows to train.";
    return;
  }

  const nFeat = Xraw[0].length;
  const minArr = Array(nFeat).fill(Infinity);
  const maxArr = Array(nFeat).fill(-Infinity);

  for (const v of Xraw) {
    for (let i = 0; i < nFeat; i++) {
      minArr[i] = Math.min(minArr[i], v[i]);
      maxArr[i] = Math.max(maxArr[i], v[i]);
    }
  }

  const Xscaled = Xraw.map(v => minMaxScale(v, minArr, maxArr));
  aiNorm = { min: minArr, max: maxArr };

  const xs = tf.tensor2d(Xscaled, [Xscaled.length, nFeat]);
  const ys = tf.tensor2d(yraw, [yraw.length, 1]);

  aiModel = tf.sequential();
  aiModel.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [nFeat] }));
  aiModel.add(tf.layers.dense({ units: 8, activation: "relu" }));
  aiModel.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  aiModel.compile({
    optimizer: tf.train.adam(0.01),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  trainStatus.textContent = "AI status: training (10 epochs)...";

  await aiModel.fit(xs, ys, {
    epochs: 10,
    batchSize: 512,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const loss = logs.loss?.toFixed(4);
        const acc = (logs.acc ?? logs.accuracy)?.toFixed(4);
        const valLoss = logs.val_loss?.toFixed(4);
        const valAcc = (logs.val_acc ?? logs.val_accuracy)?.toFixed(4);
        trainStatus.textContent =
          `AI status: epoch ${epoch + 1}/10 | loss=${loss} acc=${acc} | val_loss=${valLoss} val_acc=${valAcc}`;
        await tf.nextFrame();
      }
    }
  });

  xs.dispose();
  ys.dispose();

  trainStatus.textContent = "AI status: trained ✅ (Neural Net ready)";
  predictAndUpdate();
}

function resetAi() {
  if (aiModel) {
    aiModel.dispose();
    aiModel = null;
  }
  aiNorm = null;
  trainStatus.textContent = "AI status: not trained";
  predictAndUpdate();
}

// Events
mode.addEventListener("change", predictAndUpdate);

thallium.addEventListener("input", predictAndUpdate);
chestPain.addEventListener("input", predictAndUpdate);
cholesterol.addEventListener("input", predictAndUpdate);
bp.addEventListener("input", predictAndUpdate);
threshold.addEventListener("input", predictAndUpdate);

anginaNo.addEventListener("click", () => { exerciseAngina = 0; updateToggleButtons(); predictAndUpdate(); });
anginaYes.addEventListener("click", () => { exerciseAngina = 1; updateToggleButtons(); predictAndUpdate(); });

sex0Btn.addEventListener("click", () => { sex = 0; updateToggleButtons(); predictAndUpdate(); });
sex1Btn.addEventListener("click", () => { sex = 1; updateToggleButtons(); predictAndUpdate(); });

trainAiBtn.addEventListener("click", () => { trainAiFromCsv(); });
resetAiBtn.addEventListener("click", () => { resetAi(); });

// Init
updateEquationText();
updateToggleButtons();
predictAndUpdate();