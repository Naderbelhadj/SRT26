const pptxgen = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const FIG = '/tmp/claude-0/-home-user-SRT26/618f6450-ab76-51da-ba52-60090a9566a0/scratchpad/deckfig';
const IMG = '/home/user/SRT26/img';
const OUT = process.argv[2] || '/home/user/SRT26/defense/Belhadj_PhD_Defence.pptx';
const NOTES = JSON.parse(fs.readFileSync(path.join(__dirname, 'notes_fr.json'), 'utf8'));
function addN(slide) { slide.addNotes(NOTES[n - 1]); }

const C = {
  navy: '16324F', R: '1F4E79', Rf: 'D6E6F5', S: '1B7A4E', Sf: 'D9F0E3', O: '6B3D8C', Of: 'EADCF3',
  red: 'A32020', redf: 'F8D6D6', orange: 'B35C00', orangef: 'FCE8D2', gray: '5A6470', light: 'F3F5F8',
  white: 'FFFFFF', ink: '1F2933', ice: 'CADCFC'
};
const HF = 'Cambria', BF = 'Calibri';
const W = 13.333, H = 7.5;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Nader Belhadj';
pres.title = 'PhD defence - Deep Learning-Based Prediction of Disease Severity Using Medical Imaging';

let n = 0;
function imgSize(file) {
  // read PNG header for width/height
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, w, h, opts = {}) {
  const s = imgSize(file); const r = s.w / s.h;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = x + (w - iw) / 2, iy = opts.top ? y : y + (h - ih) / 2;
  slide.addImage({ path: file, x: ix, y: iy, w: iw, h: ih });
  return { x: ix, y: iy, w: iw, h: ih };
}
function footer(slide, dark = false) {
  slide.addText('N. Belhadj  |  PhD defence  |  ENSI, University of Manouba', {
    x: 0.6, y: H - 0.42, w: 8, h: 0.3, fontFace: BF, fontSize: 10, color: dark ? 'AFC3D6' : '8A94A0', margin: 0, isTextBox: true });
  slide.addText(String(n), { x: W - 1.4, y: H - 0.42, w: 0.8, h: 0.3, fontFace: BF, fontSize: 10,
    color: dark ? 'AFC3D6' : '8A94A0', align: 'right', margin: 0, isTextBox: true });
}
function content(title, tag, color, notes) {
  n++;
  const s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.OVAL, { x: 0.6, y: 0.42, w: 0.56, h: 0.56, fill: { color }, line: { color } });
  s.addText(tag, { x: 0.6, y: 0.42, w: 0.56, h: 0.56, fontFace: HF, fontSize: 15, bold: true, color: C.white,
    align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText(title, { x: 1.35, y: 0.3, w: 11.4, h: 0.8, fontFace: HF, fontSize: 28, bold: true, color: C.navy,
    valign: 'middle', margin: 0, isTextBox: true });
  footer(s);
  addN(s);
  return s;
}
function section(num, title, sub, color, notes) {
  n++;
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.OVAL, { x: 0.9, y: 2.55, w: 1.5, h: 1.5, fill: { color }, line: { color: C.white, width: 2 } });
  s.addText(num, { x: 0.9, y: 2.55, w: 1.5, h: 1.5, fontFace: HF, fontSize: 44, bold: true, color: C.white,
    align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText(title, { x: 2.9, y: 2.45, w: 9.6, h: 1.0, fontFace: HF, fontSize: 40, bold: true, color: C.white,
    margin: 0, valign: 'middle', isTextBox: true });
  s.addText(sub, { x: 2.9, y: 3.45, w: 9.6, h: 0.8, fontFace: BF, fontSize: 18, italic: true, color: C.ice,
    margin: 0, valign: 'top', isTextBox: true });
  footer(s, true);
  addN(s);
  return s;
}
function bullets(slide, items, x, y, w, h, size = 16, color = C.ink) {
  const arr = items.map((t, i) => {
    const last = i === items.length - 1;
    if (typeof t === 'string') return { text: t, options: { bullet: true, breakLine: !last, paraSpaceAfter: 8 } };
    return { text: t.text, options: Object.assign({ bullet: true, breakLine: !last, paraSpaceAfter: 8 }, t.o || {}) };
  });
  slide.addText(arr, { x, y, w, h, fontFace: BF, fontSize: size, color, valign: 'top', margin: 0.05, isTextBox: true });
}
function card(slide, x, y, w, h, head, body, color, fill, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  slide.addText(head, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: opts.headH || 0.45, fontFace: HF, fontSize: opts.headSize || 18,
    bold: true, color, margin: 0, valign: 'top', isTextBox: true });
  const bh = h - (opts.headH || 0.45) - 0.3;
  if (Array.isArray(body)) bullets(slide, body, x + 0.2, y + 0.2 + (opts.headH || 0.45), w - 0.4, bh, opts.size || 14);
  else slide.addText(body, { x: x + 0.2, y: y + 0.2 + (opts.headH || 0.45), w: w - 0.4, h: bh, fontFace: BF,
    fontSize: opts.size || 14, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
}
function stat(slide, x, y, w, big, label, color, size = 50) {
  slide.addText(big, { x, y, w, h: 1.0, fontFace: HF, fontSize: size, bold: true, color, margin: 0, align: 'left', isTextBox: true });
  slide.addText(label, { x, y: y + 1.0, w, h: 0.7, fontFace: BF, fontSize: 15, color: C.gray, margin: 0, valign: 'top', isTextBox: true });
}
function table(slide, rows, x, y, w, colW, opts = {}) {
  const fs_ = opts.fontSize || 13;
  const data = rows.map((r, i) => r.map((c) => {
    const cell = typeof c === 'object' ? c : { text: String(c) };
    const o = Object.assign({ fontFace: BF, fontSize: fs_, color: C.ink, valign: 'middle', margin: [3, 6, 3, 6] }, cell.options || {});
    if (i === 0) Object.assign(o, { bold: true, color: C.white, fill: { color: opts.head || C.navy } });
    else if (i % 2 === 0) Object.assign(o, { fill: { color: o.fill ? o.fill.color : C.light } });
    return { text: cell.text, options: o };
  }));
  slide.addTable(data, { x, y, w, colW, border: { type: 'solid', pt: 0.5, color: 'D5DAE1' }, rowH: opts.rowH || 0.38 });
}
function barChart(slide, x, y, w, h, labels, series, opts) {
  const data = series.map((s) => ({ name: s.name, labels, values: s.values }));
  slide.addChart(pres.charts.BAR, data, Object.assign({
    x, y, w, h, barDir: 'col', barGrouping: 'clustered', chartColors: opts.colors,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFontSize: 11, dataLabelColor: C.ink,
    dataLabelFormatCode: opts.fmt || '0.00',
    valAxisMinVal: opts.min, valAxisMaxVal: opts.max, valAxisLabelFormatCode: opts.fmt || '0.00',
    valAxisLabelColor: C.gray, catAxisLabelColor: C.ink, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    valGridLine: { color: 'E3E7EC', size: 0.5 }, catGridLine: { style: 'none' },
    showLegend: series.length > 1, legendPos: 'b', legendFontSize: 11,
    showTitle: !!opts.title, title: opts.title, titleFontSize: 14, titleColor: C.navy, titleFontFace: HF,
  }, opts.extra || {}));
}

// ---------------------------------------------------------------- 1 Title
n++;
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addImage({ path: path.join(IMG, 'logo_manouba.png'), x: 0.7, y: 0.45, h: 0.95, w: 0.95 * imgSize(path.join(IMG, 'logo_manouba.png')).w / imgSize(path.join(IMG, 'logo_manouba.png')).h });
  const le = imgSize(path.join(IMG, 'logo_ensi.png'));
  s.addImage({ path: path.join(IMG, 'logo_ensi.png'), x: W - 0.7 - 0.95 * le.w / le.h, y: 0.45, h: 0.95, w: 0.95 * le.w / le.h });
  s.addText('University of Manouba  |  National School of Computer Sciences (ENSI)  |  HANALab', {
    x: 2.2, y: 0.6, w: 8.9, h: 0.6, fontFace: BF, fontSize: 14, color: C.ice, align: 'center', margin: 0, isTextBox: true });
  s.addText('Deep Learning-Based Prediction of Disease Severity Using Medical Imaging', {
    x: 0.9, y: 1.55, w: 11.5, h: 1.6, fontFace: HF, fontSize: 34, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText('An Application to Diabetology', { x: 0.9, y: 3.3, w: 11.5, h: 0.6, fontFace: HF, fontSize: 24, italic: true,
    color: C.ice, align: 'center', margin: 0, isTextBox: true });
  const dots = [['R', C.R], ['S', C.S], ['O', C.O]];
  dots.forEach((d, i) => {
    s.addShape(pres.shapes.OVAL, { x: 5.62 + i * 0.75, y: 4.2, w: 0.55, h: 0.55, fill: { color: d[1] }, line: { color: C.white, width: 1.5 } });
    s.addText(d[0], { x: 5.62 + i * 0.75, y: 4.2, w: 0.55, h: 0.55, fontFace: HF, fontSize: 16, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  });
  s.addText('Nader BELHADJ', { x: 0.9, y: 5.0, w: 11.5, h: 0.5, fontFace: HF, fontSize: 24, bold: true, color: C.white, align: 'center', margin: 0, isTextBox: true });
  s.addText('Doctoral thesis in Computer Science (Artificial Intelligence)\nSupervisor: Prof. Lassaad Latrach  |  Co-supervisor: Prof. Ridha Ghayoula (Laval University)  |  Co-advisor: Mohamed Amine Mezghich', {
    x: 0.9, y: 5.55, w: 11.5, h: 0.9, fontFace: BF, fontSize: 14, color: C.ice, align: 'center', margin: 0, isTextBox: true });
  addN(s);
}

// ---------------------------------------------------------------- 2 Outline
{
  const s = content('Outline', 'i', C.navy,
    "[0:45 – 1:15] Le plan suit la logique de la thèse : d'abord le contexte médical expliqué simplement, puis le problème et les hypothèses, les outils utilisés, les premiers résultats, puis les trois systèmes TOPORET, DGTS et DOTS, la partie logicielle, et enfin la synthèse : ce qui est établi, ce qui ne l'est pas, et les perspectives.");
  const parts = [
    ['1', 'Context and problem', 'Diabetic retinopathy, screening, three limitations', C.navy],
    ['2', 'Hypotheses and methodology', 'H1, H2, H3; three systems; pre-specified tests', C.navy],
    ['3', 'Tools and first evidence', 'Graphs, persistent homology, ordinal regression (Ch. 2-3)', C.gray],
    ['4', 'TOPORET', 'Relations + vessel topology; tests of H1 and H2 (Ch. 4)', C.R],
    ['5', 'DGTS and DOTS', 'Inductive graph; ordinal safety; test of H3 (Ch. 5)', C.O],
    ['6', 'Engineering', 'Six-stage architecture, regulatory-oriented design (Ch. 6)', C.orange],
    ['7', 'Synthesis', 'Evidence, limits, perspectives', C.navy],
  ];
  parts.forEach((p, i) => {
    const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4;
    const x = 0.8 + col * 6.1, y = 1.55 + row * 1.25;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.05, w: 0.7, h: 0.7, fill: { color: p[3] }, line: { color: p[3] } });
    s.addText(p[0], { x, y: y + 0.05, w: 0.7, h: 0.7, fontFace: HF, fontSize: 20, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(p[1], { x: x + 0.95, y, w: 4.9, h: 0.45, fontFace: HF, fontSize: 20, bold: true, color: C.navy, margin: 0, isTextBox: true });
    s.addText(p[2], { x: x + 0.95, y: y + 0.45, w: 4.9, h: 0.4, fontFace: BF, fontSize: 14, color: C.gray, margin: 0, isTextBox: true });
  });
}

// ================================================================= PART 1
section('1', 'Context and Problem', 'Why grading diabetic retinopathy is a safety problem, not only an accuracy problem', C.navy,
  "[1:15] Première partie : le contexte. Je vais l'expliquer simplement, sans prérequis médical.");

// 4 numbers
{
  const s = content('DR: a silent, preventable cause of blindness', 'C', C.navy,
    "[1:30 – 3:00] Explication simple : le diabète abîme progressivement les petits vaisseaux sanguins, y compris ceux de la rétine, au fond de l'œil. C'est la rétinopathie diabétique. Environ 537 millions d'adultes sont diabétiques, 783 millions attendus en 2045, et environ 103 millions ont une rétinopathie. Le point essentiel : la maladie est silencieuse. Le patient voit bien jusqu'à un stade avancé, alors que le traitement est surtout efficace tôt. D'où la nécessité de dépister chaque diabétique régulièrement, en photographiant le fond d'œil.");
  stat(s, 0.8, 1.6, 3.6, '537 M', 'adults with diabetes (783 M expected by 2045)', C.navy);
  stat(s, 4.75, 1.6, 3.6, '103 M', 'people with diabetic retinopathy (DR)', C.red);
  stat(s, 8.7, 1.6, 3.9, 'Silent', 'vision is preserved until the disease is advanced; treatment works best early', C.O);
  card(s, 0.8, 4.05, 11.75, 2.55, 'What it means in practice',
    ['Every diabetic patient should have the retina photographed regularly (fundus photograph).',
     'Each image is graded by an expert on the five-level international ICDR scale.',
     'The number of images now far exceeds the number of available graders, especially where ophthalmologists are scarce: automated grading is the natural answer.'],
    C.navy, C.light, { size: 16 });
}

// 5 grades
{
  const s = content('What the grader sees: the five ICDR grades', 'C', C.navy,
    "[3:00 – 4:30] Voici les cinq grades de l'échelle internationale ICDR, de 0 (pas de rétinopathie) à 4 (rétinopathie proliférante). Les signes apparaissent dans les vaisseaux : micro-anévrismes, hémorragies, puis, au stade proliférant, de nouveaux vaisseaux anormaux. Ce qui compte pour la sécurité : les grades 3 et 4 doivent être orientés vers l'ophtalmologiste en quelques semaines, voire en urgence. Les grades sont ordonnés : confondre 0 et 4 est bien plus grave que confondre 1 et 2. C'est une idée clé de la thèse.");
  const g = [['0', 'No DR', 'No lesion', 'Annual recall'], ['1', 'Mild', 'Microaneurysms only', '12 months'],
    ['2', 'Moderate', 'Haemorrhages, exudates', '6 months'], ['3', 'Severe', '4-2-1 rule', '2-4 weeks'], ['4', 'Proliferative', 'New vessels', 'Emergency']];
  g.forEach((r, i) => {
    const x = 0.7 + i * 2.43;
    fitImage(s, path.join(IMG, `fundus_grade${r[0]}.png`), x, 1.45, 2.2, 2.2);
    const col = i >= 3 ? C.red : C.navy;
    s.addText(`Grade ${r[0]}  ${r[1]}`, { x, y: 3.8, w: 2.2, h: 0.45, fontFace: HF, fontSize: 16, bold: true, color: col, align: 'center', margin: 0, isTextBox: true });
    s.addText(r[2], { x, y: 4.25, w: 2.2, h: 0.4, fontFace: BF, fontSize: 13, color: C.ink, align: 'center', margin: 0, isTextBox: true });
    s.addText(r[3], { x, y: 4.65, w: 2.2, h: 0.4, fontFace: BF, fontSize: 13, italic: true, color: i >= 3 ? C.red : C.gray, align: 'center', margin: 0, isTextBox: true });
  });
  card(s, 0.7, 5.35, 11.9, 1.3, 'Two facts that drive the thesis',
    'The grades are ordered (confusing 0 with 4 is far worse than 1 with 2), and Severe (3) and Proliferative (4) cases must be referred quickly.',
    C.navy, C.light, { size: 16 });
}

// 6 safety
{
  const s = content('Why accuracy is not enough: missed severe cases', 'C', C.red,
    "[4:30 – 6:00] Un système peut avoir une bonne précision globale et pourtant rater des cas graves. La mesure de sécurité de la thèse est le Sev-NR : la proportion d'images sévères classées « pas de RD » ou « légère », donc non orientées. Exemple illustratif : un programme de 500 000 patients par an, dont environ 25 000 cas sévères. Avec un Sev-NR de 5,8 % – celui d'un CNN de référence sur Messidor-2 dans cette thèse – environ 1 450 cas sévères ne seraient pas orientés chaque année. C'est une projection sous hypothèses, pas un résultat clinique, mais elle montre pourquoi on mesure la sécurité à côté de la précision.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 1.5, w: 5.6, h: 2.4, rectRadius: 0.08, fill: { color: C.redf }, line: { color: C.redf } });
  s.addText([
    { text: 'Sev-NR', options: { bold: true, fontSize: 26, color: C.red, breakLine: true } },
    { text: 'share of Severe (Grade 3) images predicted as No DR or Mild, i.e. not referred', options: { fontSize: 16, color: C.ink, breakLine: true } },
    { text: 'PDR-NR: the same for Grade 4', options: { fontSize: 14, color: C.gray } }],
    { x: 1.05, y: 1.65, w: 5.1, h: 2.1, fontFace: BF, margin: 0, valign: 'top', isTextBox: true });
  s.addText('Illustrative projection', { x: 7.0, y: 1.5, w: 5.5, h: 0.45, fontFace: HF, fontSize: 18, bold: true, color: C.navy, margin: 0, isTextBox: true });
  stat(s, 7.0, 2.0, 2.7, '25,000', 'severe cases among 500,000 screened patients / year', C.navy, 38);
  stat(s, 9.9, 2.0, 2.7, '1,450', 'left unreferred at Sev-NR = 5.8% (CNN baseline, Messidor-2)', C.red, 38);
  card(s, 0.8, 4.35, 11.75, 2.3, 'Consequence for the thesis',
    ['Safety metrics (Sev-NR, PDR-NR) are reported next to accuracy (QWK), and checked on data from another country.',
     'The projection is a calculation under assumed prevalence, not an estimate of clinical impact.'],
    C.red, C.light, { size: 16 });
}

// 7 limitations
{
  const s = content('Three limitations of representative systems', 'C', C.navy,
    "[6:00 – 7:30] En étudiant les systèmes représentatifs – Gulshan 2016, CABNet, RETFound – on observe trois limites architecturales. L1 : chaque image est classée seule ; des patients qui se ressemblent ne s'informent jamais. L2 : les réseaux voient la texture locale, mais ne mesurent pas explicitement la structure globale de l'arbre vasculaire, ses composantes et ses boucles. L3 : les grades sont traités comme des catégories sans ordre et le taux de cas sévères non orientés est rarement mesuré. Précision importante : c'est une remarque d'architecture, pas un jugement sur toute la littérature ; plusieurs systèmes ont été validés prospectivement.");
  const L = [['L1', 'No relational context', 'Each image is graded in isolation: similar patients, who often share a grade, never inform one another.', C.R, C.Rf],
    ['L2', 'No vascular topology', 'Networks see local texture, not the global structure of the vessel tree (components, loops), which changes with the disease.', C.S, C.Sf],
    ['L3', 'No safety-oriented grading', 'Grades treated as unordered; the rate of unreferred severe cases is rarely measured, let alone externally.', C.O, C.Of]];
  L.forEach((l, i) => {
    const x = 0.8 + i * 3.97;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: 3.75, h: 2.95, rectRadius: 0.08, fill: { color: l[4] }, line: { color: l[4] } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.7, w: 0.75, h: 0.75, fill: { color: l[3] }, line: { color: l[3] } });
    s.addText(l[0], { x: x + 0.25, y: 1.7, w: 0.75, h: 0.75, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(l[1], { x: x + 1.15, y: 1.7, w: 2.45, h: 0.75, fontFace: HF, fontSize: 18, bold: true, color: l[3], valign: 'middle', margin: 0, isTextBox: true });
    s.addText(l[2], { x: x + 0.25, y: 2.65, w: 3.25, h: 2.0, fontFace: BF, fontSize: 15, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
  table(s, [['System', 'Year', 'Method', 'L1', 'L2', 'L3'],
    ['Gulshan et al.', '2016', 'Deep CNN (Inception-v3)', '✗', '✗', '✗'],
    ['CABNet', '2021', 'Category attention on a CNN', '✗', '✗', '✗'],
    ['RETFound', '2023', 'ViT-L/16, retinal pre-training', '✗', '✗', '✗']],
    0.8, 4.75, 11.75, [2.6, 1.0, 5.15, 1.0, 1.0, 1.0], { fontSize: 13, rowH: 0.4 });
}

// ================================================================= PART 2
section('2', 'Hypotheses and Methodology', 'One objective, three hypotheses, three dedicated systems', C.navy,
  "[7:30] Deuxième partie : ce que la thèse cherche à tester, et comment.");

{
  const s = content('Objective and research hypotheses', 'H', C.navy,
    "[7:45 – 9:15] Objectif unique : exploiter les relations, la topologie et l'ordinalité pour améliorer le grading. Trois hypothèses, chacune liée à une limite. H1 : un graphe de population apporte une information au-delà d'un classifieur image par image. H2 : l'homologie persistante du réseau vasculaire est statistiquement associée à la sévérité. H3 : une chaîne ordinale satisfait des objectifs de sécurité fixés à l'avance – QWK ≥ 0,80, Sev-NR et PDR-NR ≤ 2 %, aucune confusion 0↔4 – en interne et sur données externes (H3a), alors qu'aucune de ses variantes réduites n'y parvient (H3b). Point méthodologique important : pour H1 et H2, le test, le seuil et la taille d'effet ont été fixés avant l'expérience.");
  s.addText('Objective: exploit relations, topology and ordinality to improve DR grading', {
    x: 0.8, y: 1.35, w: 11.75, h: 0.55, fontFace: HF, fontSize: 20, italic: true, color: C.navy, margin: 0, isTextBox: true });
  const Hs = [['H1', 'Relational', 'A population graph adds information beyond a per-image classifier.', 'Paired t-test over folds (pre-specified)', C.R, C.Rf],
    ['H2', 'Structural', 'Persistent homology of the vessel network is statistically associated with severity.', 'One-way ANOVA (pre-specified)', C.S, C.Sf],
    ['H3', 'Ordinal safety', 'An ordinal pipeline satisfies pre-set screening targets internally and externally (H3a); no reduced variant does (H3b).', 'QWK ≥ 0.80; Sev-NR, PDR-NR ≤ 2%; no 0↔4 confusion', C.O, C.Of]];
  Hs.forEach((h, i) => {
    const y = 2.1 + i * 1.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y, w: 11.75, h: 1.3, rectRadius: 0.08, fill: { color: h[5] }, line: { color: h[5] } });
    s.addShape(pres.shapes.OVAL, { x: 1.0, y: y + 0.25, w: 0.8, h: 0.8, fill: { color: h[4] }, line: { color: h[4] } });
    s.addText(h[0], { x: 1.0, y: y + 0.25, w: 0.8, h: 0.8, fontFace: HF, fontSize: 20, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(h[1], { x: 2.0, y: y + 0.12, w: 2.2, h: 1.05, fontFace: HF, fontSize: 18, bold: true, color: h[4], valign: 'middle', margin: 0, isTextBox: true });
    s.addText(h[2], { x: 4.2, y: y + 0.12, w: 5.0, h: 1.05, fontFace: BF, fontSize: 15, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
    s.addText(h[3], { x: 9.4, y: y + 0.12, w: 3.0, h: 1.05, fontFace: BF, fontSize: 13, italic: true, color: C.gray, valign: 'middle', margin: 0, isTextBox: true });
  });
}

{
  const s = content('A three-axis research framework, not a single model', 'M', C.navy,
    "[9:15 – 10:45] Choix méthodologique essentiel : la thèse propose un cadre de recherche à trois axes, pas un modèle unique. Chaque axe est porté par un système dédié, pour mesurer sa contribution séparément. TOPORET teste relations et topologie vasculaire (H1, H2), en mode transductif. DGTS reprend ces deux ingrédients en mode inductif, avec un régulariseur topologique dans l'espace des caractéristiques, et l'évalue sur EyePACS. DOTS ajoute l'ordinalité et est testé contre les objectifs de sécurité (H3), sur Messidor-2. Les trois systèmes ne fournissent donc pas la même forme de preuve – et je le dis explicitement. Tous les jeux de données sont publics et les graines fixées.");
  table(s, [['', 'TOPORET (Ch. 4)', 'DGTS (Ch. 5)', 'DOTS (Ch. 5)'],
    [{ text: 'Axes', options: { bold: true } }, 'Relational + structural', 'Relational + structural', 'Ordinal safety'],
    [{ text: 'Topology', options: { bold: true } }, 'Vessel skeleton (6 persistence statistics)', 'Feature-space neighbourhoods (training regulariser)', 'None'],
    [{ text: 'Training data', options: { bold: true } }, 'Each benchmark separately', 'Fused corpus, 5,859 images', 'Fused corpus, 5,859 images'],
    [{ text: 'Test images in graph', options: { bold: true } }, 'Yes (unlabelled nodes)', 'No (graph of training fold)', 'No graph'],
    [{ text: 'Protocol', options: { bold: true } }, 'Transductive', 'Inductive', 'Per-image'],
    [{ text: 'External evaluation', options: { bold: true } }, 'None (within benchmark)', 'EyePACS, zero-shot', 'Messidor-2'],
    [{ text: 'Kind of evidence', options: { bold: true } }, { text: 'Transductive (H1, H2)', options: { color: C.R, bold: true } }, { text: 'Inductive, external', options: { color: C.S, bold: true } }, { text: 'Screening-oriented, point estimate (H3)', options: { color: C.O, bold: true } }]],
    0.8, 1.45, 11.75, [2.75, 3.0, 3.0, 3.0], { fontSize: 14, rowH: 0.56 });
  s.addText('Pre-specified tests for H1 and H2: test, decision threshold and effect size fixed before the experiment.', {
    x: 0.8, y: 6.2, w: 11.75, h: 0.45, fontFace: BF, fontSize: 15, italic: true, color: C.gray, margin: 0, isTextBox: true });
}

{
  const s = content('A common six-stage framework', 'M', C.navy,
    "[10:45 – 11:30] Tous les systèmes reposent sur une chaîne en six étapes à interfaces typées : prétraitement et squelette vasculaire, backbone CNN, homologie persistante, fusion, graphe de population, puis tête de classement. Chaque étape appartient à un axe (couleurs). Les interfaces typées permettent de tester chaque étape séparément, ce que j'exploite au chapitre 6 pour la vérification logicielle.");
  fitImage(s, path.join(FIG, 'fig_pipeline.png'), 0.8, 1.6, 11.75, 3.0);
  card(s, 0.8, 4.9, 5.7, 1.7, 'Colour code', ['Blue: relational axis (L1)  |  Green: structural axis (L2)', 'Purple: ordinal axis (L3)  |  Orange: shared fusion'], C.navy, C.light, { size: 14 });
  card(s, 6.85, 4.9, 5.7, 1.7, 'Why typed stages', ['Each stage can be tested on its own (Chapter 6)', 'The same stages are reused by TOPORET, DGTS and DOTS'], C.navy, C.light, { size: 14 });
}

// ================================================================= PART 3
section('3', 'Tools and First Evidence', 'Graph neural networks, persistent homology, ordinal regression (Chapters 2-3)', C.gray,
  "[11:30] Troisième partie : les trois outils, puis les premiers résultats qui ont motivé TOPORET.");

{
  const s = content('Three tools, explained simply', 'T', C.gray,
    "[11:45 – 13:45] Trois outils, expliqués simplement. (1) Graphe de population et GraphSAGE : chaque image est un nœud ; on relie les images qui se ressemblent ; le réseau agrège l'information des voisins – l'idée est qu'un cas ressemble à ses voisins. (2) Homologie persistante : on regarde le squelette des vaisseaux comme un nuage de points et on fait grossir des boules autour de chaque point ; on compte les composantes (H0) et les boucles (H1) qui apparaissent et disparaissent. Les boucles qui persistent longtemps sont les vraies structures, les autres sont du bruit. (3) CORAL : au lieu de prédire 5 classes sans ordre, on répond à 4 questions binaires emboîtées « le grade est-il ≥ 1 ? ≥ 2 ? ≥ 3 ? ≥ 4 ? », ce qui respecte l'ordre des grades.");
  const cols = [['Population graph + GraphSAGE', 'fig_messagepassing.png', 'Each image is a node; similar images are linked; each node aggregates its neighbours.', C.R],
    ['Persistent homology', 'fig_filtration.png', 'Grow balls around skeleton points; count components (H0) and loops (H1) that persist across scales.', C.S],
    ['CORAL ordinal regression', 'fig_coral.png', 'Four nested binary questions: is the grade ≥ 1, ≥ 2, ≥ 3, ≥ 4? The order of grades is built in.', C.O]];
  cols.forEach((c, i) => {
    const x = 0.8 + i * 3.97;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 3.75, h: 5.2, rectRadius: 0.08, fill: { color: C.light }, line: { color: C.light } });
    s.addText(c[0], { x: x + 0.2, y: 1.6, w: 3.35, h: 0.7, fontFace: HF, fontSize: 18, bold: true, color: c[3], margin: 0, valign: 'middle', isTextBox: true });
    fitImage(s, path.join(FIG, c[1]), x + 0.2, 2.4, 3.35, 2.1);
    s.addText(c[2], { x: x + 0.2, y: 4.7, w: 3.35, h: 1.8, fontFace: BF, fontSize: 15, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
}

{
  const s = content('First evidence: structure carries information', 'F', C.gray,
    "[13:45 – 15:30] Avant de construire TOPORET, j'ai accumulé des indices. F1 : le gabarit de graphe a d'abord été testé hors médecine – orchestration de microservices et détection d'anomalies – avec un F1 de 94,4 à 97,0 % dans l'étude de conférence. Mais la réévaluation sur dix exécutions montre honnêtement que les invariants ajoutés apportent interprétabilité et robustesse à la dérive, pas un gain de précision significatif. F2 : sur APTOS en binaire, le modèle hybride CNN-GNN avec géométrie et topologie gagne +7,0 points (95,1 %), plus que la somme des gains individuels, sur un seul découpage. F3 : la structure vasculaire globale est discriminante, confirmée par trois méthodes ; SAMF-GB, un modèle sur CPU, gagne +0,042 de QWK face à EfficientNet-B3 sur GPU. Ce n'est pas une comparaison d'architectures à armes égales, mais cela montre que la structure contient de l'information.");
  const F = [['F1', 'Graph template outside medicine', 'F1 94.4-97.0% (single run); ten runs: interpretability and robustness to drift, no significant accuracy gain.'],
    ['F2', 'Relations and topology: complementary', 'Hybrid CNN-GNN on APTOS 2019 (binary): 95.1% accuracy, +7.0 pp over the CNN, single split.'],
    ['F3', 'Vessel structure is discriminative', 'Three methods agree; SAMF-GB: +0.042 QWK over EfficientNet-B3, CPU classifier about 13x cheaper.']];
  F.forEach((f, i) => {
    const y = 1.5 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: y + 0.1, w: 0.75, h: 0.75, fill: { color: C.gray }, line: { color: C.gray } });
    s.addText(f[0], { x: 0.8, y: y + 0.1, w: 0.75, h: 0.75, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(f[1], { x: 1.75, y, w: 5.4, h: 0.45, fontFace: HF, fontSize: 15, bold: true, color: C.navy, margin: 0, isTextBox: true });
    s.addText(f[2], { x: 1.75, y: y + 0.45, w: 5.2, h: 0.8, fontFace: BF, fontSize: 14, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
  fitImage(s, path.join(IMG, 'hybrid_confusion_5class.png'), 7.3, 1.45, 5.25, 4.0);
  s.addText('Hybrid CNN-GNN, five-class confusion matrix (APTOS 2019)', { x: 7.3, y: 5.5, w: 5.25, h: 0.35, fontFace: BF, fontSize: 11, italic: true, color: C.gray, align: 'center', margin: 0, isTextBox: true });
  s.addText('Take-away: worth testing a population graph with topology inside the similarity (Chapter 4).', {
    x: 0.8, y: 5.95, w: 11.75, h: 0.6, fontFace: HF, fontSize: 17, italic: true, color: C.navy, margin: 0, isTextBox: true });
}

// ================================================================= PART 4 TOPORET
section('4', 'TOPORET', 'A dual-similarity population graph: relations + vessel topology (Chapter 4)', C.R,
  "[15:30] Quatrième partie : TOPORET, le cœur des hypothèses H1 et H2.");

{
  const s = content('TOPORET: a dual-similarity population graph', 'R', C.R,
    "[15:45 – 17:15] TOPORET relie les images dans un graphe de population. Deux images sont proches si elles se ressemblent pour le CNN (similarité cosinus) ET si leurs réseaux vasculaires ont une topologie similaire (terme exponentiel sur la distance des descripteurs topologiques). α = 0,6 et τ = 0,65 ont été choisis par grid search sur la validation. La topologie joue un double rôle : dans les arêtes et dans les caractéristiques des nœuds (1 536 dimensions CNN + 6 topologiques = 1 542). Protocole, à dire clairement : validation croisée 5 plis, graine 42, au niveau image, transductive – les images de test sont des nœuds non étiquetés du graphe. À ma connaissance, les descripteurs de persistance du squelette vasculaire n'avaient pas été utilisés dans les arêtes d'un graphe de population.");
  fitImage(s, path.join(FIG, 'fig_toporet.png'), 0.8, 1.4, 11.75, 2.7);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.8, y: 4.3, w: 6.6, h: 2.35, rectRadius: 0.08, fill: { color: C.Rf }, line: { color: C.Rf } });
  s.addText([
    { text: 'Dual similarity', options: { bold: true, fontSize: 18, color: C.R, fontFace: HF, breakLine: true } },
    { text: 'S(i,j) = α · cos(CNN_i, CNN_j) + (1 − α) · exp(−d_ij / d_max)', options: { fontSize: 15, color: C.ink, fontFace: 'Cambria Math', breakLine: true } },
    { text: 'd_ij: distance of the 6-D topological vectors; edge if S > τ', options: { fontSize: 14, color: C.gray, breakLine: true } },
    { text: 'α = 0.6, τ = 0.65 (validation grid search)', options: { fontSize: 14, color: C.gray } }],
    { x: 1.05, y: 4.45, w: 6.2, h: 2.1, fontFace: BF, margin: 0, valign: 'top', isTextBox: true });
  card(s, 7.7, 4.3, 4.85, 2.35, 'Protocol (stated plainly)', ['Five-fold CV, seed 42, image level', 'Transductive: test images are unlabelled nodes', 'Three benchmarks, each evaluated within itself'], C.R, C.light, { size: 14 });
}

{
  const s = content('Vessel topology with persistent homology', 'S', C.S,
    "[17:15 – 18:30] Concrètement : on améliore le contraste avec CLAHE, on extrait le squelette des vaisseaux, on prend les points de bifurcation comme nuage de points, puis on calcule la filtration de Vietoris-Rips avec GUDHI. On résume les diagrammes par six nombres : persistance totale, entropie et nombre de composantes longues pour H0, et la même chose pour H1 (les boucles). Une barre est « longue » si elle dépasse 5 % de la plus longue barre. Précision honnête : la similarité compare ces six statistiques, pas les diagrammes complets – c'est un résumé grossier de la topologie.");
  fitImage(s, path.join(IMG, 'vessel_skeleton_persistence.png'), 0.8, 1.45, 6.4, 5.1);
  card(s, 7.5, 1.45, 5.05, 2.55, 'Six descriptors per image', ['H0 (components): total persistence, entropy, number of long-lived bars', 'H1 (loops): total persistence, entropy, number of long-lived bars'], C.S, C.Sf, { size: 15 });
  card(s, 7.5, 4.2, 5.05, 2.4, 'Pipeline', ['CLAHE, skeleton, branch points', 'Vietoris-Rips filtration (GUDHI), edge ≤ 100 px', 'Long-lived bar: > 5% of the longest bar'], C.S, C.light, { size: 15 });
}

{
  const s = content('Results on three benchmarks', 'R', C.R,
    "[18:30 – 19:30] Sur les trois bases, TOPORET améliore toutes les métriques : +0,04 de QWK sur Kaggle DR, +0,07 sur Messidor-2, +0,04 sur APTOS, et +1,5 à +2,3 points de précision. Le gain est le plus grand sur Messidor-2, dont l'acquisition standardisée donne des squelettes plus propres. Attention : ces chiffres absolus sont obtenus en transductif au niveau image ; ils ne se comparent pas directement aux systèmes publiés. Ce qui est valide, c'est la comparaison appariée avec le même CNN, les mêmes caractéristiques, les mêmes plis.");
  barChart(s, 0.8, 1.4, 7.2, 5.2, ['Kaggle DR', 'Messidor-2', 'APTOS 2019'],
    [{ name: 'CNN baseline', values: [0.84, 0.83, 0.82] }, { name: 'TOPORET', values: [0.88, 0.90, 0.86] }],
    { colors: ['A7B4C2', C.R], min: 0.75, max: 0.95, title: 'QWK (mean over five folds)' });
  table(s, [['Dataset', 'Accuracy (%)', 'ΔQWK'],
    ['Kaggle DR', '94.0 → 95.5', '+0.04'], ['Messidor-2', '93.8 → 96.1', '+0.07'], ['APTOS 2019', '92.9 → 94.6', '+0.04']],
    8.4, 1.6, 4.15, [1.55, 1.6, 1.0], { fontSize: 14, rowH: 0.45, head: C.R });
  card(s, 8.4, 3.7, 4.15, 2.9, 'Read with care', ['Transductive, image-level folds', 'Not comparable with published absolute figures', 'Valid: paired comparison, same features and folds'], C.red, C.redf, { size: 14 });
}

{
  const s = content('H1: converging but limited evidence', 'H1', C.R,
    "[19:30 – 21:30] Le test pré-spécifié : t-test apparié sur les 5 plis, seuil |t4| > 2,78. Sur Kaggle DR, t4 = 3,82, p = 0,019 : H1 passe, et aussi sur Messidor-2 et APTOS. Mais 5 plis, c'est peu : j'ai ajouté un bootstrap (IC [+0,021 ; +0,059], exclut zéro) et un test de permutation exact (p = 1/32, le minimum possible : le gain est positif dans chaque pli). Et comme les plis partagent leurs données d'entraînement, j'ai appliqué la correction de Nadeau-Bengio : aucune base seule ne reste sous 0,05 (p = 0,070 ; 0,058 ; 0,088), mais les trois bases indépendantes combinées par la méthode de Fisher donnent p = 0,015. Verdict : des preuves convergentes mais limitées en faveur de H1, pas une preuve générale. Je préfère le dire moi-même.");
  table(s, [['Dataset', 'ΔQWK', 't4', 'p (pre-specified)', 'Corrected t (Nadeau-Bengio)', 'Corrected p'],
    ['Kaggle DR (primary)', '+0.04', '3.82', '0.019', '2.45', '0.070'],
    ['Messidor-2', '+0.07', '4.1', '0.015', '2.63', '0.058'],
    ['APTOS 2019', '+0.04', '3.5', '0.025', '2.25', '0.088'],
    [{ text: 'Three benchmarks combined (Fisher)', options: { bold: true } }, '', '', '', '', { text: '0.015', options: { bold: true, color: C.R } }]],
    0.8, 1.45, 11.75, [3.2, 1.2, 1.1, 2.0, 2.65, 1.6], { fontSize: 14, rowH: 0.45, head: C.R });
  const b = [['Bootstrap CI', '[+0.021, +0.059]', 'excludes zero'], ['Sign-flip test', 'p = 1/32', 'gain > 0 in every fold'], ['Effect size', 'd_z = 1.71', 'consistency, not size']];
  b.forEach((x, i) => {
    const cx = 0.8 + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: 4.0, w: 2.9, h: 1.6, rectRadius: 0.08, fill: { color: C.Rf }, line: { color: C.Rf } });
    s.addText(x[0], { x: cx + 0.15, y: 4.1, w: 2.6, h: 0.4, fontFace: BF, fontSize: 13, color: C.gray, margin: 0, isTextBox: true });
    s.addText(x[1], { x: cx + 0.15, y: 4.5, w: 2.6, h: 0.6, fontFace: HF, fontSize: 18, bold: true, color: C.R, margin: 0, isTextBox: true });
    s.addText(x[2], { x: cx + 0.15, y: 5.1, w: 2.6, h: 0.4, fontFace: BF, fontSize: 13, italic: true, color: C.ink, margin: 0, isTextBox: true });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.2, y: 4.0, w: 2.35, h: 1.6, rectRadius: 0.08, fill: { color: C.R }, line: { color: C.R } });
  s.addText('Converging but limited evidence', { x: 10.3, y: 4.05, w: 2.15, h: 1.5, fontFace: HF, fontSize: 17, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText('Five folds give four degrees of freedom; the evidence supports H1 under the evaluated, transductive conditions, not a general advantage of population graphs.', {
    x: 0.8, y: 5.85, w: 11.75, h: 0.75, fontFace: BF, fontSize: 15, italic: true, color: C.gray, margin: 0, isTextBox: true });
}

{
  const s = content('Where the gain comes from: ablation', 'R', C.R,
    "[21:30 – 22:30] L'ablation sur Kaggle DR isole chaque composant. Les descripteurs topologiques seuls sur les nœuds (B) : +0,02. Le graphe avec des caractéristiques CNN seules, la topologie uniquement dans les arêtes (C) : +0,03, le plus gros contributeur. Le modèle complet (D) : +0,04. Les contributions sont complémentaires et à peu près additives. La matrice de confusion montre que le gain le plus net est à la frontière Légère–Modérée, la plus ambiguë aussi pour les humains. Les tests de l'ablation ne sont pas corrigés du chevauchement des plis – je le signale.");
  barChart(s, 0.8, 1.4, 6.6, 5.2, ['(A) CNN only', '(B) + TDA features', '(C) + graph (TDA in edges)', '(D) Full TOPORET'],
    [{ name: 'QWK', values: [0.84, 0.86, 0.87, 0.88] }], { colors: [C.R], min: 0.80, max: 0.90, title: 'QWK on Kaggle DR (five folds)' });
  fitImage(s, path.join(IMG, 'toporet_confusion.png'), 7.7, 1.4, 4.85, 3.9);
  s.addText('Largest gain at the Mild-Moderate boundary', { x: 7.7, y: 5.35, w: 4.85, h: 0.35, fontFace: BF, fontSize: 12, italic: true, color: C.gray, align: 'center', margin: 0, isTextBox: true });
  s.addText('The graph is the larger contributor; topology is complementary.', { x: 7.7, y: 5.8, w: 4.85, h: 0.8, fontFace: HF, fontSize: 16, bold: true, color: C.R, margin: 0, isTextBox: true });
}

{
  const s = content('H2: a small but robust statistical association', 'H2', C.S,
    "[22:30 – 24:00] H2 : une ANOVA pré-spécifiée de la persistance totale H1 sur les cinq grades, sur les 35 126 images de Kaggle DR. F = 84,3, p < 0,001, environ 23 fois la valeur critique. Le seuil de décision 0,005 est inférieur au niveau de Bonferroni pour six descripteurs. Et même si l'échantillon effectif était divisé par deux, F resterait autour de 42. MAIS la taille d'effet est petite : η² ≈ 0,01, le grade explique environ 1 % de la variance. Donc : association statistique robuste, pas une causalité, pas un biomarqueur, pas une signification biologique démontrée. Un lien avec la néovascularisation est plausible mais non vérifié. C'est cohérent avec l'ablation : la topologie est une information complémentaire.");
  stat(s, 0.8, 1.5, 3.6, 'F = 84.3', 'one-way ANOVA of total H1 persistence across 5 grades; p < 0.001', C.S, 42);
  stat(s, 4.75, 1.5, 3.6, 'η² ≈ 0.01', 'grade explains about 1% of the variance: a small effect', C.red, 42);
  stat(s, 8.7, 1.5, 3.85, 'F ≈ 42', 'if the effective sample size were halved (both eyes per patient)', C.navy, 42);
  card(s, 0.8, 3.55, 5.75, 3.05, 'What H2 shows', ['A statistical association on 35,126 images', 'Monotone increase of H1 persistence with grade', 'α = 0.005 < Bonferroni level 0.05/6', 'Complementary to visual features (+0.02 QWK)'], C.S, C.Sf, { size: 15 });
  card(s, 6.8, 3.55, 5.75, 3.05, 'What H2 does not show', ['No causality, no demonstrated biological meaning', 'Not a biomarker: "severity-associated topological feature"', 'Link with neovascularisation plausible, not verified', 'Tested on Kaggle DR only'], C.red, C.redf, { size: 15 });
}

{
  const s = content('TOPORET: threats to validity, stated openly', 'R', C.R,
    "[24:00 – 25:00] Les limites de TOPORET, que j'expose moi-même. Transductif : les caractéristiques des images de test, pas leurs grades, influencent leurs voisins pendant l'entraînement ; ce n'est pas équivalent à une évaluation strictement inductive et indépendante du patient. Œil adelphe : Kaggle DR et Messidor-2 contiennent les deux yeux, une image test peut être reliée à l'autre œil du même patient. Deux faits limitent ce risque : le gain est aussi observé sur APTOS, une image par patient, et DGTS montre un gain du graphe en mode inductif. La réévaluation au niveau patient est la première perspective : le code publié n'a besoin que d'un découpage groupé.");
  const T = [['Internal', 'Transductive protocol: test images are unlabelled nodes; not a strictly inductive, patient-independent evaluation.'],
    ['Statistical', 'Five folds, one seed: four degrees of freedom; corrected significance only for the three benchmarks combined.'],
    ['External', 'Both eyes in Kaggle DR and Messidor-2: possible fellow-eye information. Gain also on APTOS (one image per patient).'],
    ['Clinical', 'Retrospective public data only; no reader study, no prospective evaluation.']];
  T.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    card(s, 0.8 + col * 6.0, 1.45 + row * 2.25, 5.75, 2.05, t[0] + ' validity', t[1], C.R, C.light, { size: 15 });
  });
  s.addText('Answer in the thesis: inductive evidence from DGTS; patient-level re-evaluation is the first perspective.', {
    x: 0.8, y: 6.05, w: 11.75, h: 0.55, fontFace: HF, fontSize: 16, italic: true, color: C.navy, margin: 0, isTextBox: true });
}

// ================================================================= PART 5 DGTS & DOTS
section('5', 'DGTS and DOTS', 'Inductive graph learning and ordinal, safety-oriented grading (Chapter 5)', C.O,
  "[25:00] Cinquième partie : les deux systèmes entraînés sur un corpus multi-source, et le test de H3.");

{
  const s = content('Two systems on one multi-source corpus', 'D', C.O,
    "[25:15 – 26:30] TOPORET laissait trois lacunes : entraînement sur une seule source, perte qui ignore l'ordre, sécurité non mesurée. DGTS et DOTS partagent le même corpus fusionné : APTOS 2019 (3 662 images) + KaggleDR-F1 (2 197) = 5 859 images, 309 sévères et 472 PDR. Ici le découpage image est aussi un découpage patient. Honnêteté sur la provenance : KaggleDR-F1 contient exactement 60 % des effectifs d'APTOS dans chaque grade ; un chevauchement n'est pas exclu, donc la validation croisée est lue comme une preuve interne multi-source, possiblement optimiste. Les conclusions reposent donc sur les évaluations externes : EyePACS pour DGTS (35 126 images, vérifié sans doublon) et Messidor-2 pour DOTS (1 744 images, France).");
  fitImage(s, path.join(FIG, 'fig_dots.png'), 0.8, 1.4, 7.4, 3.3);
  table(s, [['Data', 'Images', 'Role'], ['APTOS 2019 (India)', '3,662', 'Training'], ['KaggleDR-F1', '2,197', 'Training'],
    [{ text: 'Fused corpus', options: { bold: true } }, { text: '5,859', options: { bold: true } }, '5-fold CV, seed 42'],
    ['EyePACS', '35,126', 'External (DGTS)'], ['Messidor-2 (France)', '1,744', 'External (DOTS)']],
    8.5, 1.45, 4.05, [1.8, 0.95, 1.3], { fontSize: 13, rowH: 0.42, head: C.O });
  card(s, 0.8, 4.95, 11.75, 1.7, 'Dataset provenance and independence',
    'KaggleDR-F1 has exactly 60% of the APTOS count in every grade: overlap not excluded. Cross-validation = internal multi-source evidence; the external evaluations carry the greater weight.',
    C.red, C.redf, { size: 15 });
}

{
  const s = content('DGTS: an inductive, topology-regularised graph', 'S', C.S,
    "[26:30 – 28:00] DGTS : EfficientNet-B4 gelé, graphe k-plus-proches-voisins cosinus avec k = 8 construit uniquement sur le pli d'entraînement – donc inductif – et GraphSAGE à deux couches. En parallèle, une branche d'homologie persistante décrit la forme du voisinage de chaque nœud dans l'espace des caractéristiques. Point subtil, souvent demandé : à l'inférence, le descripteur vaut zéro ; la topologie agit comme un régulariseur d'entraînement, via les gradients. Tous les chiffres sont mesurés dans ce réglage. Et ce n'est pas la topologie des vaisseaux. L'ablation (sur APTOS) : le graphe apporte +0,018 de QWK (A2→A3), le régulariseur +0,030 (A2→A4), les deux ensemble +0,052, la fusion +0,011. Wilcoxon DGTS contre CNN : p = 0,031.");
  barChart(s, 0.8, 1.4, 7.3, 5.25, ['A1 CNN', 'A2 + focal', 'A3 + graph', 'A4 + PH', 'A5 graph+PH', 'A6 DGTS (fused)', 'A7 EMD loss'],
    [{ name: 'QWK', values: [0.858, 0.878, 0.896, 0.908, 0.930, 0.941, 0.921] }], { colors: [C.S], min: 0.80, max: 0.96, fmt: '0.000', title: 'DGTS ablation, QWK (A1-A5 on APTOS 2019)' });
  card(s, 8.4, 1.45, 4.15, 2.75, 'Architecture', ['EfficientNet-B4 (frozen)', 'Cosine k-NN graph, k = 8, training fold only', 'GraphSAGE 1792 → 512 → 256', 'PH of neighbourhoods: training-time regulariser'], C.S, C.Sf, { size: 13 });
  card(s, 8.4, 4.4, 4.15, 2.25, 'Key facts', ['Graph +0.018, PH +0.030 QWK', 'At inference the PH input is zero', 'Feature-space, not vascular, topology'], C.S, C.light, { size: 13 });
}

{
  const s = content('DGTS on external data: accuracy is not safety', 'S', C.S,
    "[28:00 – 29:00] Appliqué sans réentraînement aux 35 126 images EyePACS, DGTS atteint un QWK de 0,925, soit une baisse de seulement 0,016 par rapport à la validation croisée (0,941). C'est une bonne généralisation. Mais le Sev-NR externe est d'environ 2,1 %, légèrement au-dessus de l'objectif de 2 %. Leçon principale de la thèse : une bonne précision ne garantit pas la sécurité ; il faut les mesurer séparément. C'est ce qui motive DOTS. Côté pratique : 18,3 ms par image sur un GPU T4.");
  stat(s, 0.8, 1.6, 3.6, '0.941', 'QWK, five-fold cross-validation (±0.008)', C.S);
  stat(s, 4.75, 1.6, 3.6, '0.925', 'QWK zero-shot on 35,126 EyePACS images (−0.016)', C.S);
  stat(s, 8.7, 1.6, 3.85, '≈ 2.1%', 'Sev-NR on EyePACS: slightly above the 2% target', C.red);
  card(s, 0.8, 3.9, 11.75, 1.6, 'Lesson', 'Good generalisation in QWK does not by itself guarantee the safety criterion: accuracy and safety must be measured separately. This motivates DOTS.', C.red, C.redf, { size: 17 });
  s.addText('Inference: 18.3 ms per image on one T4 GPU (14.1 ms for EfficientNet-B4 alone).', { x: 0.8, y: 5.85, w: 11.75, h: 0.5, fontFace: BF, fontSize: 15, italic: true, color: C.gray, margin: 0, isTextBox: true });
}

{
  const s = content('Screening-oriented targets of H3', 'O', C.O,
    "[29:00 – 29:45] Les objectifs de H3, fixés à l'avance : QWK ≥ 0,80, Sev-NR ≤ 2 %, PDR-NR ≤ 2 %, aucune confusion 0↔4. Ils viennent de l'article DOTS et s'inspirent de la pratique du dépistage (essai pivot d'IA autonome, programme national anglais). Ce ne sont pas des seuils réglementaires. Et la définition est volontairement explicite : une image sévère prédite « Modérée » compte comme orientée, ce qui correspond à un parcours où le grade Modéré est déjà orientable ; une définition plus stricte donnerait des taux plus élevés.");
  table(s, [['Criterion', 'Threshold', 'Rationale'],
    ['Quadratic weighted kappa (QWK)', '≥ 0.80', 'Substantial agreement with the reference grades'],
    ['Severe non-referral rate (Sev-NR)', '≤ 2%', 'Missed urgent referrals, Grade 3'],
    ['PDR non-referral rate (PDR-NR)', '≤ 2%', 'Missed urgent referrals, Grade 4'],
    ['Grade 0↔4 confusions', '0%', 'Catastrophic ordinal errors']],
    0.8, 1.5, 11.75, [4.3, 1.8, 5.65], { fontSize: 15, rowH: 0.55, head: C.O });
  card(s, 0.8, 4.55, 5.75, 2.05, 'Where they come from', ['DOTS article; informed by the autonomous-AI pivotal trial and national screening programmes', 'Evaluation targets, not regulatory thresholds'], C.O, C.Of, { size: 14 });
  card(s, 6.8, 4.55, 5.75, 2.05, 'Definition, stated explicitly', ['Severe predicted as Moderate counts as referred', 'A stricter definition (≥ Grade 3) would give higher rates'], C.red, C.redf, { size: 14 });
}

{
  const s = content('DOTS: ordinal, safety-oriented grading', 'O', C.O,
    "[29:45 – 31:15] DOTS : EfficientNet-B3 fine-tuné, une projection, un encodeur de raffinement (FRE), une tête ordinale CORAL et une augmentation au test (5 transformations). Honnêteté sur le FRE : il a la forme d'un bloc transformer mais opère sur une séquence de longueur un, donc l'attention ne fait rien ; c'est en pratique un MLP résiduel normalisé, et je ne revendique aucun bénéfice de l'attention. L'ablation, avec le même budget de 10 époques : 0,903 → TTA 0,918 → GraphSAGE 0,928 → FRE à la place du graphe 0,937 → CORAL 0,943. Le graphe aide ici aussi (cohérent avec H1), mais le FRE fait mieux à moindre coût. CORAL apporte le plus petit gain de QWK, mais c'est la sécurité qui compte, comme on va le voir.");
  barChart(s, 0.8, 1.4, 7.3, 5.25, ['EfficientNet-B3', '+ TTA ×5', '+ GraphSAGE', 'FRE replacing graph', '+ CORAL (DOTS)'],
    [{ name: 'QWK', values: [0.903, 0.918, 0.928, 0.937, 0.943] }], { colors: [C.O], min: 0.88, max: 0.95, fmt: '0.000', title: 'Progressive ablation, QWK (5-fold CV, 10 epochs)' });
  card(s, 8.4, 1.45, 4.15, 2.6, 'Pipeline', ['EfficientNet-B3, fine-tuned, 320×320', 'Projection + FRE (residual MLP)', 'CORAL head: 4 rank logits', 'Test-time augmentation ×5'], C.O, C.Of, { size: 13 });
  card(s, 8.4, 4.25, 4.15, 2.4, 'Honest note', ['FRE runs on a sequence of length one: attention is inactive', 'No attention-based benefit is claimed'], C.red, C.redf, { size: 13 });
}

{
  const s = content('H3a: targets satisfied at the point estimate', 'H3', C.O,
    "[31:15 – 32:45] Résultats de DOTS. En interne, sur 5 859 prédictions hors pli : QWK 0,953, Sev-NR 0,97 % (3 images sur 309), PDR-NR 1,27 % (6 sur 472), aucune confusion 0↔4. Sur Messidor-2, jamais vu à l'entraînement : QWK 0,931, Sev-NR 1,4 %, PDR-NR 1,9 %. Les performances observées satisfont donc tous les objectifs en estimation ponctuelle. Mais – et c'est essentiel – avec 309 images sévères, l'intervalle exact du Sev-NR interne est [0,2 % ; 2,8 %], il dépasse 2 %. La sécurité n'est donc pas démontrée. Il faudrait environ 650 images sévères pour que la borne supérieure passe sous 2 %.");
  table(s, [['Criterion', 'Target', 'Internal (N = 5,859)', 'Messidor-2 (N = 1,744)'],
    ['QWK', '≥ 0.80', '0.953 [0.944, 0.961]', '0.931 [0.919, 0.942]'],
    ['Sev-NR', '≤ 2%', '0.97% (3/309) [0.2%, 2.8%]', '1.4%'],
    ['PDR-NR', '≤ 2%', '1.27% (6/472) [0.5%, 2.7%]', '1.9%'],
    ['Grade 0↔4 confusions', '0%', '0%', '0%'],
    [{ text: 'Verdict', options: { bold: true } }, '', { text: 'satisfied at the point estimate', options: { bold: true, color: C.O } }, { text: 'satisfied at the point estimate', options: { bold: true, color: C.O } }]],
    0.8, 1.45, 11.75, [3.0, 1.5, 3.75, 3.5], { fontSize: 15, rowH: 0.5, head: C.O });
  card(s, 0.8, 4.75, 7.6, 1.85, 'Statistical uncertainty', 'With 309 severe images the internal interval [0.2%, 2.8%] crosses 2%; external intervals cannot be computed from the published counts. Safety is not demonstrated.', C.red, C.redf, { size: 15 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.7, y: 4.75, w: 3.85, h: 1.85, rectRadius: 0.08, fill: { color: C.O }, line: { color: C.O } });
  s.addText([{ text: '≈ 650', options: { fontSize: 36, bold: true, fontFace: HF, breakLine: true } }, { text: 'severe images needed for an upper bound below 2%', options: { fontSize: 13 } }],
    { x: 8.85, y: 4.85, w: 3.55, h: 1.65, fontFace: BF, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}

{
  const s = content('H3b: only the full pipeline meets the target', 'H3', C.O,
    "[32:45 – 34:00] H3b sur Messidor-2 : chaque variante réduite échoue sur le Sev-NR : 5,8 % pour le CNN de base, 4,1 % avec GraphSAGE, 3,7 % avec CORAL seul, 2,9 % avec TTA seule. Seule la chaîne complète descend à 1,4 %. Les barres montrent le QWK, les lignes les taux de non-orientation. Enseignement : le composant qui améliore le plus la sécurité n'est pas celui qui améliore le plus le QWK. Limite : la conclusion vaut pour les variantes de cette ablation, pas pour tout système sans l'un de ces composants.");
  fitImage(s, path.join(FIG, 'fig_progressive.png'), 0.8, 1.35, 8.2, 4.4);
  card(s, 9.3, 1.45, 3.25, 4.3, 'Messidor-2 Sev-NR', ['Baseline 5.8%', '+ GraphSAGE 4.1%', '+ CORAL only 3.7%', '+ TTA only 2.9%', 'DOTS complete 1.4%'], C.O, C.Of, { size: 15 });
  s.addText('The component that improves safety most is not the one that improves QWK most. Claim limited to the variants of this ablation.', {
    x: 0.8, y: 5.95, w: 11.75, h: 0.7, fontFace: HF, fontSize: 16, italic: true, color: C.navy, margin: 0, isTextBox: true });
}

{
  const s = content('How DOTS errs: in the safe direction', 'O', C.O,
    "[34:00 – 34:45] La structure des erreurs compte autant que leur nombre. Sur 597 erreurs, 82,9 % sont entre grades adjacents, aucune ne traverse quatre grades. La confusion dominante est Légère → Modérée, une sur-orientation, pas une sous-orientation. Les images sévères ont une faible précision mais un rappel de 0,770 : DOTS prédit « Sévère » plus souvent qu'il ne faudrait, ce qui est la bonne direction pour l'orientation. Calibration : l'ECE passe de 0,067 à 0,031 par temperature scaling, sans changer le QWK. McNemar DOTS contre CNN : p = 0,013.");
  stat(s, 0.8, 1.6, 3.6, '82.9%', 'of the 597 errors are between adjacent grades; none spans four grades', C.O);
  stat(s, 4.75, 1.6, 3.6, '0.770', 'recall of Severe images: over-prediction, the safe direction', C.O);
  stat(s, 8.7, 1.6, 3.85, '0.031', 'five-grade ECE after temperature scaling (from 0.067), QWK unchanged', C.navy);
  card(s, 0.8, 4.0, 11.75, 2.1, 'Reading', ['Dominant confusion Mild → Moderate: an over-referral rather than an under-referral', 'Referral decision: AUC 0.971 [0.963, 0.979]; ECE 0.041', 'McNemar test, DOTS vs CNN baseline on paired predictions: p = 0.013'], C.O, C.light, { size: 16 });
}

// ================================================================= PART 6
section('6', 'Engineering', 'Six-stage architecture and regulatory-oriented design (Chapter 6)', C.orange,
  "[34:45] Sixième partie, brièvement : l'ingénierie logicielle.");

{
  const s = content('Designed with regulatory considerations in mind', 'E', C.orange,
    "[35:00 – 36:15] Le chapitre 6 relie l'architecture aux exigences d'IEC 62304 (cycle de vie et vérification unitaire), d'ISO 14971 (gestion des risques, AMDEC sur six dangers) et de l'EU AI Act (système à haut risque, traçabilité, supervision humaine). C'est un exercice d'ingénierie, pas une évaluation de conformité, ni un marquage CE, ni une autorisation FDA. Le tableau d'état le dit clairement : les pipelines sont implémentés en code de recherche et évalués rétrospectivement ; le plan de vérification est spécifié ; l'adaptateur FHIR est conçu mais pas implémenté ; pas de déploiement ni de certification.");
  table(s, [['Component', 'Status'],
    ['Grading pipelines (TOPORET, DGTS, DOTS)', 'Implemented as research code; evaluated retrospectively'],
    ['Stage-level verification plan (IEC 62304)', 'Specified'],
    ['Risk analysis (FMEA, ISO 14971)', 'Performed at design level'],
    ['Audit trail and Article 13 template (EU AI Act)', 'Designed'],
    ['Automatic deferral of uncertain images', { text: 'Not implemented', options: { color: C.red, bold: true } }],
    ['FHIR adapter and hospital integration', { text: 'Designed, not implemented', options: { color: C.red, bold: true } }],
    ['Clinical deployment, conformity assessment, CE / FDA', { text: 'Not performed', options: { color: C.red, bold: true } }]],
    0.8, 1.45, 11.75, [6.0, 5.75], { fontSize: 15, rowH: 0.52, head: C.orange });
  s.addText('An engineering-oriented mapping exercise, not a conformity assessment, clinical evaluation or certification.', {
    x: 0.8, y: 5.95, w: 11.75, h: 0.6, fontFace: HF, fontSize: 16, italic: true, color: C.orange, margin: 0, isTextBox: true });
}

// ================================================================= PART 7
section('7', 'Synthesis', 'What is shown, what is supported, what remains to be done', C.navy,
  "[36:15] Dernière partie : la synthèse.");

{
  const s = content('Evidence architecture of the thesis', 'Σ', C.navy,
    "[36:30 – 37:15] Cette figure résume l'ensemble : un objectif commun, trois systèmes qui apportent chacun un type de preuve différent – transductive pour TOPORET, inductive et externe pour DGTS, orientée dépistage pour DOTS –, toute cette preuve est rétrospective, et elle s'arrête avant l'évaluation prospective. En bas, en rouge, ce qui n'est pas encore évalué.");
  fitImage(s, path.join(FIG, 'fig_evidence.png'), 0.8, 1.35, 11.75, 5.3);
}

{
  const s = content('Shown, supported, and not claimed', 'Σ', C.navy,
    "[37:15 – 38:30] Je distingue trois niveaux. Montré directement : les quantités mesurées – gain de TOPORET dans chaque pli des trois bases, gains du graphe et du régulariseur dans DGTS, QWK externe de DGTS, estimations ponctuelles de DOTS qui satisfont les objectifs, échec des quatre variantes réduites. Soutenu sans être établi de façon concluante : que l'information relationnelle améliore le grading, que la topologie porte une information liée à la sévérité, que la chaîne ordinale est assez sûre pour le dépistage. Non revendiqué : sécurité clinique, conformité réglementaire, prédiction longitudinale, supériorité générale des graphes.");
  table(s, [['Claim', 'Evidence', 'Level', 'Main limitation'],
    ['Relational information improves grading', 'H1 (TOPORET); DGTS +0.018 QWK, inductive', 'Moderate', 'Five folds; corrected p only combined'],
    ['Topology is associated with severity', 'H2: F = 84.3, p < 0.001', 'Moderate', 'η² ≈ 0.01; association only'],
    ['Topological regularisation helps', 'DGTS ablation +0.030 QWK', 'Moderate', 'Feature-space topology'],
    ['DGTS keeps its QWK externally', 'EyePACS zero-shot 0.925', 'Moderate-strong', 'Sev-NR ≈ 2.1%'],
    ['DOTS satisfies the targets (H3a)', 'Internal + Messidor-2, point estimates', 'Moderate', 'Interval [0.2%, 2.8%]'],
    ['No reduced variant suffices (H3b)', 'Four variants fail on Messidor-2', 'Moderate', 'One ablation specification'],
    [{ text: 'Clinical safety, regulatory compliance', options: { color: C.red, bold: true } }, '-', { text: 'Not established', options: { color: C.red, bold: true } }, 'No prospective study; no conformity assessment']],
    0.8, 1.45, 11.75, [3.55, 3.6, 1.7, 2.9], { fontSize: 13, rowH: 0.52 });
  s.addText('Central contribution: a structured body of retrospective evidence on how relational, topological and ordinal information can complement deep visual representations.', {
    x: 0.8, y: 5.85, w: 11.75, h: 0.75, fontFace: HF, fontSize: 16, italic: true, color: C.navy, margin: 0, isTextBox: true });
}

{
  const s = content('Limitations and perspectives', 'P', C.navy,
    "[38:30 – 39:30] Les limites définissent directement les perspectives, dans l'ordre de coût. D'abord réévaluer TOPORET au niveau patient avec un graphe par pli ; puis répéter les expériences sur plusieurs graines ; ajouter de nouvelles bases externes avec assez de cas sévères ; construire un modèle unifié – graphe, topologie vasculaire, tête CORAL – avec un mécanisme d'abstention fondé sur les probabilités calibrées de DOTS ; puis une étude de lecteurs, une étude prospective pré-enregistrée, dimensionnée sur environ 1 240 images sévères pour 80 % de puissance, une vraie validation multi-site ou fédérée, et enfin la validation clinique au sens du règlement sur les dispositifs médicaux.");
  const P = ['Patient-level split for TOPORET (grouped folds, per-fold graph)', 'Multi-seed validation', 'New external datasets (≈ 650 severe images)', 'Unified three-axis model with deferral',
    'Reader study (graders with and without the system)', 'Prospective study (≈ 1,240 severe images, 80% power)', 'True multi-site and federated validation', 'Clinical validation (MDR)'];
  P.forEach((p, i) => {
    const col = i < 4 ? 0 : 1, row = i % 4;
    const x = 0.8 + col * 6.0, y = 1.5 + row * 1.2;
    const color = i < 3 ? C.R : (i < 4 ? C.O : C.S);
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.08, w: 0.65, h: 0.65, fill: { color }, line: { color } });
    s.addText(String(i + 1), { x, y: y + 0.08, w: 0.65, h: 0.65, fontFace: HF, fontSize: 18, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(p, { x: x + 0.85, y, w: 4.95, h: 0.8, fontFace: BF, fontSize: 16, color: C.ink, valign: 'middle', margin: 0, isTextBox: true });
  });
  s.addText('Blue: statistical limits, no new infrastructure  |  Purple: unified model  |  Green: clinical translation', {
    x: 0.8, y: 6.2, w: 11.75, h: 0.4, fontFace: BF, fontSize: 13, italic: true, color: C.gray, margin: 0, isTextBox: true });
}

{
  const s = content('Scientific output', 'P', C.navy,
    "[39:30 – 39:45] Treize publications soutiennent la thèse : neuf acceptées ou publiées – sept articles de conférence (ICAART 2025 et 2026, ICPRAM, ENASE, MIUA, IJCNN) et deux chapitres de livre LNAI chez Springer – et quatre articles de revue, dont un en révision à PLOS ONE et trois en cours d'évaluation.");
  stat(s, 0.8, 1.5, 3.6, '13', 'publications supporting the thesis', C.navy);
  stat(s, 4.75, 1.5, 3.6, '9', 'accepted or published (7 conference papers, 2 book chapters)', C.S);
  stat(s, 8.7, 1.5, 3.85, '4', 'journal articles: 1 under revision (PLOS ONE), 3 under review', C.O);
  table(s, [['Chapter', 'Content', 'Publications'],
    ['3', 'Graph template outside medicine; structural evidence on DR', 'P1-P4, P7, B1, B2, J2'],
    ['4', 'TOPORET; tests of H1 and H2', 'J3 (PLOS ONE), P5 (ENASE), P6 (MIUA)'],
    ['5', 'DGTS and DOTS; H3', 'J1, J4'],
    ['6', 'Software architecture, regulatory-oriented design', 'P5']],
    0.8, 3.7, 11.75, [1.3, 6.25, 4.2], { fontSize: 14, rowH: 0.5 });
}

// ---------------------------------------------------------------- Conclusion
n++;
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText('Conclusion', { x: 0.9, y: 0.6, w: 11.5, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true });
  const K = [['R', 'Relations', 'Evidence supporting H1: consistent gains, limited by five folds; inductive confirmation with DGTS.', C.R],
    ['S', 'Topology', 'A small but robust statistical association (H2); a useful regulariser in DGTS; not a biomarker.', C.S],
    ['O', 'Ordinality', 'DOTS satisfies the screening targets at the point estimate, internally and on Messidor-2; safety not yet demonstrated.', C.O]];
  K.forEach((k, i) => {
    const y = 1.75 + i * 1.3;
    s.addShape(pres.shapes.OVAL, { x: 0.9, y, w: 0.85, h: 0.85, fill: { color: k[3] }, line: { color: C.white, width: 1.5 } });
    s.addText(k[0], { x: 0.9, y, w: 0.85, h: 0.85, fontFace: HF, fontSize: 22, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(k[1], { x: 2.0, y, w: 2.3, h: 0.85, fontFace: HF, fontSize: 22, bold: true, color: C.white, valign: 'middle', margin: 0, isTextBox: true });
    s.addText(k[2], { x: 4.3, y, w: 8.1, h: 0.85, fontFace: BF, fontSize: 16, color: C.ice, valign: 'middle', margin: 0, isTextBox: true });
  });
  s.addText('Relational, topological and ordinal information can complement deep visual representations for DR grading: here is the evidence, and here is exactly where it stops.', {
    x: 0.9, y: 5.7, w: 11.5, h: 0.8, fontFace: HF, fontSize: 18, italic: true, color: C.white, margin: 0, isTextBox: true });
  s.addText('Thank you for your attention', { x: 0.9, y: 6.55, w: 11.5, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: C.ice, margin: 0, isTextBox: true });
  addN(s);
}

pres.writeFile({ fileName: OUT }).then(() => console.log('wrote', OUT, 'slides', n));
