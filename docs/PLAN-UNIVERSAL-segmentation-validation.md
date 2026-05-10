# Segmentasyon Doğrulama (İkili 3D Viewer + WT/TC/ET Metrikleri)

## Goal
Sonuç sayfasında, kullanıcının yüklediği BraTS2021 örneği için modelin ürettiği segmentasyon ile uzman tarafından üretilmiş ground-truth segmentasyonu yan yana 3D olarak göstermek ve aralarındaki uyumu Dice + Hausdorff95 metrikleriyle WT/TC/ET bölgeleri için sayısal olarak raporlamak. Yüklenen dosya BraTS pattern'ına uymuyorsa (ör. gerçek hasta MR'ı), GT viewer ve metrik kartı zarif şekilde gizlenir, sadece tek viewer gösterilir.

## Scope

### In Scope
- Yüklenen modalite dosya adlarından subject ID çıkarma (`BraTS2021_XXXXX_<modality>.nii.gz` pattern'ı)
- Subject ID üzerinden `data/raw/BraTS2021/{sid}/{sid}_seg.nii.gz` dosyasının otomatik tespit edilmesi
- Backend'de Dice ve Hausdorff95 metriklerinin WT (whole tumor), TC (tumor core), ET (enhancing tumor) bölgeleri için hesaplanması
- Backend results endpoint'inin GT dosyasını ve metrikleri istemciye sunması
- ResultsPage'de iki ayrı 3D viewer (model çıktısı | GT) ve altında metrik özet kartı
- GT bulunamadığında zarif düşüş (sadece tek viewer + bilgi notu)
- Hesaplama backend tarafında inference sonrası tek seferde yapılır, metadata.json'a yazılır

### Out of Scope
- Kullanıcının elle GT dosyası yüklemesi
- BraTS dışı dataset'ler için GT kaynak entegrasyonu
- Klinik onay / regulatory iddiası
- Voxel-bazlı diff overlay (sadece iki ayrı viewer + metrikler)
- Eğitim sırasında validation metrik hesaplaması (training pipeline)
- Çoklu çalışma (multiple runs) karşılaştırması veya skor geçmişi

## Constraints
- Etiket uzayı sabit: GT ve model çıktısı her ikisi de `{0, 1, 2, 4}`, shape `(240, 240, 155)` (zaten doğrulandı)
- BraTS challenge metric tanımı: WT = labels ∈ {1,2,4}, TC = labels ∈ {1,4}, ET = label = 4
- Hausdorff95 hesaplaması voxel-bazlı; spacing affine'den türetilebilir ama prototip için voxel cinsinden kabul edilebilir
- `numpy`, `scipy` ve `nibabel` zaten yüklü; `medpy` opsiyonel — yoksa scipy.ndimage ile manuel HD95 implementasyonu yapılmalı
- Backend GT dosyasını okurken yalnızca `data/raw/BraTS2021/` altındaki güvenli yolları okumalı (path traversal koruması)
- Frontend mevcut `SegmentationViewer` bileşenini ikinci viewer için yeniden kullanmalı; yeni bir 3D engine eklenmemeli

## Success Criteria
- [ ] BraTS dataset'inden 4 modaliteyi yükleyip segmentasyon çalıştırınca sonuç sayfasında iki 3D viewer ve metrik kartı görünür
- [ ] Metrik kartı WT, TC, ET için ayrı ayrı Dice ve HD95 değerlerini gösterir
- [ ] BraTS pattern'ına uymayan dosya yüklendiğinde GT viewer ve metrik kartı görünmez, "GT bulunamadı" bilgisi gösterilir, model viewer normal çalışır
- [ ] Metrikler backend tarafında inference sonrası hesaplanır ve metadata.json'a yazılır
- [ ] BraTS2021_00000 örneği için Dice WT ≥ 0.85 (önceki spike değeri 0.938'di)
- [ ] Tüm yeni Python modülleri için pytest birim testleri yazılır ve geçer

## Context
- Current state:
  - `src/inference/predictor.py` `BrainTumorPredictor.predict()` BraTS native space `{0,1,2,4}` mask döner
  - `backend/app/tasks/segmentation_task.py` predict çıktısını NIfTI'a kaydedip metadata.json üretir
  - `backend/app/services/file_manager.py` upload session yönetimi yapar (`identify_modalities`, `validate_nifti_files`)
  - `backend/app/api/routes/results.py` task durumunu ve dosyalarını sunar
  - `frontend/src/pages/ResultsPage.tsx` tek bir SegmentationViewer ve VolumeReport bileşeni gösterir
  - `frontend/src/components/SegmentationViewer/SegmentationViewer.tsx` 3D render yapar
- Known opportunity: Etiket eşleşmesi ve shape uyumu doğrulandı (BRATS_TO_TRAIN_LABELS roundtrip lossless, predictor çıktısı 240×240×155 ve BraTS labels). Model çıktısı GT ile ekstra dönüştürmeye gerek olmadan voxel-bazlı karşılaştırılabilir.
- Assumptions:
  - Yüklenen dosya adları BraTS pattern'ında ise GT mutlaka `data/raw/BraTS2021/{sid}/` altında bulunur
  - Tüm 4 modalite aynı subject ID'sini paylaşmalı; karışık subject yüklendiğinde "GT yok" durumuna düşülür
  - Frontend React + TypeScript + Vite stack'i kullanmaya devam ediyor

---

## Task List

### TASK-01 — BraTS subject ID extractor
Priority: P0
Model Tier: T1 - Fast
Depends on: none

Why:
GT lookup'ın temel mekanizması. Modalite dosya adlarından tutarlı subject ID üretemezsek hiçbir doğrulama akışı çalışmaz.

Inputs:
- `backend/app/services/file_manager.py` (mevcut `identify_modalities` döndürdüğü modality→path map)
- BraTS dosya adı pattern'ı: `BraTS2021_<5-digit>_<modality>.nii.gz`

Targets:
- Yeni: `backend/app/services/brats_lookup.py`

Implementation Notes:
- `extract_subject_id(modality_paths: dict[str, Path]) -> str | None` saf fonksiyon yaz
- Regex: `^(BraTS2021_\d{5})_(t1|t1ce|t2|flair)\.nii\.gz$` (case-insensitive)
- 4 modalitenin tümü aynı subject ID'sini vermezse `None` döndür
- Hiçbir dosya pattern'a uymazsa `None` döndür
- Dosya adı kullanılır, dosya yolu üzerinden değil (upload session içindeki orijinal isim)
- FileManager `identify_modalities` upload edilen orijinal dosya adını koruyor mu? Korumuyorsa öncelikle upload aşamasında orijinal adı saklayacak şekilde küçük bir genişletme gerekir — bunu TASK-02'de ele al

Done When:
- [ ] `extract_subject_id` fonksiyonu yazıldı, sadece tüm modaliteler aynı `BraTS2021_XXXXX` ID'sini paylaşırsa o ID'yi döner
- [ ] Birim testler yazıldı: tutarlı durum, karışık ID'ler, pattern dışı isimler, eksik modalite

Verification:
- Automated: `pytest tests/backend/services/test_brats_lookup.py -v`
- Manual: Yok

---

### TASK-02 — Upload session'da orijinal dosya adı koruma
Priority: P0
Model Tier: T2 - Balanced
Depends on: TASK-01

Why:
Subject ID extraction kullanıcının yüklediği orijinal dosya adına dayanır. FileManager şu an dosya adlarını yeniden yazıyorsa (`{modality}.nii.gz` gibi standartlaştırma yapıyorsa) orijinal isim kaybolur.

Inputs:
- `backend/app/services/file_manager.py`
- `backend/app/api/routes/upload.py`

Targets:
- `backend/app/services/file_manager.py` (gerekirse upload metadata yazma)
- `backend/app/api/routes/upload.py` (gerekirse orijinal isim aktarımı)

Implementation Notes:
- Önce mevcut kodu oku: `identify_modalities` upload edilmiş dosyaları nasıl haritalıyor? Standart bir isim yazıyorsa session klasörüne küçük `upload_manifest.json` ekle: `{"modality": "<original filename>"}`
- Eğer dosyalar zaten orijinal isimle saklanıyorsa bu task no-op'tur — yine de doğrulama yap ve manifest'i sadeleştirici garanti olarak yaz
- Manifest'i okuyacak yardımcı: `FileManager.get_original_filenames(session_id) -> dict[str, str]`
- Geriye dönük uyumluluk: manifest yoksa mevcut dosya adlarını fallback olarak döndür

Done When:
- [ ] Upload edilen orijinal dosya adları session yaşam süresince erişilebilir
- [ ] `FileManager.get_original_filenames(session_id)` metodu eklendi ve test edildi
- [ ] Mevcut upload akışı bozulmadı (regression test geçer)

Verification:
- Automated: `pytest tests/backend/services/test_file_manager.py -v`
- Manual: Bir upload yapıp `tmp/uploads/<session_id>/` altındaki manifesti incele

---

### TASK-03 — GT path resolver
Priority: P0
Model Tier: T1 - Fast
Depends on: TASK-01, TASK-02

Why:
Subject ID elde edildikten sonra disk üzerinde güvenli bir şekilde GT dosyasını bulmak gerekir.

Inputs:
- TASK-01'in `extract_subject_id` fonksiyonu
- `backend/app/core/config.py` `Settings` (raw data root yolu için)

Targets:
- `backend/app/services/brats_lookup.py` (TASK-01 ile aynı dosya, fonksiyon eklenir)
- `backend/app/core/config.py` (gerekirse `brats_raw_root` ayarı eklenir)

Implementation Notes:
- `resolve_gt_path(subject_id: str, raw_root: Path) -> Path | None`
- Tam yol: `raw_root / "BraTS2021" / subject_id / f"{subject_id}_seg.nii.gz"`
- Path traversal koruması: `subject_id`'nin TASK-01 regex'ini geçtiğini bir kez daha doğrula (`re.fullmatch`); aksi halde `None` döndür
- Dosya gerçekten var mı (`Path.is_file()`) kontrol et; yoksa `None`
- `Settings.brats_raw_root` config alanı ekle, default `data/raw`
- Resolved path mutlaka `raw_root.resolve()` altında kalmalı (symlink/relative paths'a karşı)

Done When:
- [ ] `resolve_gt_path` fonksiyonu eklendi
- [ ] Path traversal testleri yazıldı (örn. `subject_id="../../../etc/passwd"`)
- [ ] Gerçek BraTS örnekleri için doğru yolu döner, eksik dosya için None döner

Verification:
- Automated: `pytest tests/backend/services/test_brats_lookup.py::test_resolve_gt_path -v`
- Manual: BraTS2021_00000 ve var olmayan ID için fonksiyonu Python REPL'de çağır

---

### TASK-04 — Segmentasyon doğrulama metrik modülü
Priority: P0
Model Tier: T3 - Power
Depends on: none

Why:
Dice ve Hausdorff95 hesaplamaları doğru implementasyon gerektirir; Dice formülü standart ama HD95 voxel sınır mesafesi içerir ve yanlış implementasyon yanıltıcı sonuç verir. BraTS challenge metric tanımları (WT/TC/ET birleşim grupları) literatürle uyumlu olmalı.

Inputs:
- `numpy`, `scipy.ndimage` (zaten yüklü)
- BraTS metric tanımları:
  - WT (Whole Tumor) = mask ∈ {1, 2, 4}
  - TC (Tumor Core) = mask ∈ {1, 4}
  - ET (Enhancing Tumor) = mask == 4

Targets:
- Yeni: `src/inference/validation.py`

Implementation Notes:
- `binarize_region(mask: np.ndarray, region: str) -> np.ndarray` — `region ∈ {"WT","TC","ET"}` için boolean mask döner
- `dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float`:
  - Formül: `2 * |P ∩ G| / (|P| + |G|)`
  - Edge case: hem pred hem gt boşsa Dice = 1.0; sadece biri boşsa 0.0
- `hausdorff95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: tuple[float,float,float] = (1,1,1)) -> float`:
  - `medpy` mevcutsa `medpy.metric.binary.hd95` kullan
  - Yoksa: `scipy.ndimage.distance_transform_edt` ile çift yönlü mesafeleri hesapla, 95. persentili al:
    1. `dt_gt = distance_transform_edt(~gt_bin, sampling=spacing)` → pred sınır voxel'lerinin GT'ye uzaklığı
    2. `dt_pred = distance_transform_edt(~pred_bin, sampling=spacing)`
    3. Sınır voxel'leri için `morphology.binary_erosion` ile sınırı çıkar (`pred_bin & ~erode(pred_bin)`)
    4. Birleşik mesafe dağılımının 95. persentili
  - Edge case: hem pred hem gt boşsa HD95 = 0.0; sadece biri boşsa `float("inf")` (NaN değil) döndür ve dokümante et
  - Shape mismatch'te `ValueError` fırlat
- `compute_validation_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1,1,1)) -> dict[str, dict[str, float]]`:
  - Çıktı: `{"WT": {"dice": 0.93, "hd95": 4.2}, "TC": {...}, "ET": {...}}`
  - Tüm değerler JSON-serializable float (NaN/Inf'i string'e çevirme — sentinel değer kullan: -1.0 ve metadata flag)
- Tüm fonksiyonlar saf, side-effect'siz; loglamayı arayan tarafa bırak

Done When:
- [ ] `src/inference/validation.py` modülü tamamlandı
- [ ] Birim testler: bilinen sentetik mask çiftleri için Dice ve HD95 değerleri elle hesaplananla eşleşir
- [ ] Edge case testleri: boş maskeler, identik maskeler, shape mismatch
- [ ] Mevcut Predictor ile entegrasyon olmadan bağımsız çalışır

Verification:
- Automated: `pytest tests/inference/test_validation.py -v`
- Manual: BraTS2021_00000 üzerinde elle çağırıp Dice WT > 0.85 doğrula

---

### TASK-05 — Metriği inference task'ına entegre et
Priority: P0
Model Tier: T2 - Balanced
Depends on: TASK-03, TASK-04

Why:
Metrikler tek truth source olarak backend'de inference sonrası hesaplanmalı; sonuç metadata.json'a yazılarak tüm istemcilere tek API'den sunulur.

Inputs:
- TASK-03: `extract_subject_id`, `resolve_gt_path`
- TASK-04: `compute_validation_metrics`
- `backend/app/tasks/segmentation_task.py` `execute_segmentation`

Targets:
- `backend/app/tasks/segmentation_task.py`

Implementation Notes:
- `execute_segmentation` içinde, postprocessing'den sonra, save_nifti'den önce şu adımları ekle:
  1. `original_filenames = manager.get_original_filenames(session_id)` (TASK-02)
  2. `subject_id = extract_subject_id(original_filenames)`
  3. `gt_path = resolve_gt_path(subject_id, settings.brats_raw_root) if subject_id else None`
  4. `validation = None`
  5. `if gt_path is not None:` blok:
     - `gt_array, gt_affine = load_nifti(gt_path)`
     - Shape kontrolü: `gt_array.shape == cleaned_mask.shape` değilse warn log + `validation = None`, exception fırlatma
     - Spacing'i affine'den hesapla: `spacing = compute_voxel_spacing(affine)` (yardımcı yoksa `np.linalg.norm(affine[:3, :3], axis=0)`)
     - `validation_metrics = compute_validation_metrics(cleaned_mask, gt_array, spacing)`
     - GT dosyasını result_dir'a kopyala: `shutil.copyfile(gt_path, result_dir / "ground_truth.nii.gz")`
     - `validation = {"subject_id": subject_id, "metrics": validation_metrics, "gt_filename": "ground_truth.nii.gz"}`
- `build_metadata` imzasına `validation: dict | None = None` ekle, varsa metadata['validation']'a yaz
- Progress update'leri: validation hesaplama için `progress=90, step="validation"` ekle
- Hata durumlarında validation hesaplaması inference'i öldürmemeli — `try/except` ile sarıla, başarısızsa warn log
- GT dosyası result_dir'a `ground_truth.nii.gz` adıyla kopyalanır (orijinal subject ID'yi metadata'da tut)

Done When:
- [ ] BraTS2021_00000 örneği yüklenip task çalıştırıldığında result_dir'da `ground_truth.nii.gz` ve metadata.json'da `validation` alanı oluşur
- [ ] Pattern dışı isimle yapılan upload'da validation alanı `None` kalır, segmentation normal tamamlanır
- [ ] Metrik hesaplama hatası inference'i kıramaz (try/except)
- [ ] Birim test: mock predictor ile validation alanının doğru doldurulduğu doğrulanır

Verification:
- Automated: `pytest tests/backend/tasks/test_segmentation_task.py -v`
- Manual: Gerçek bir BraTS örneği yükle → result_dir'a bak (`tmp/results/<task_id>/metadata.json`)

---

### TASK-06 — Results endpoint'i validation alanını ve GT dosyasını sunsun
Priority: P0
Model Tier: T1 - Fast
Depends on: TASK-05

Why:
Frontend, GT dosyasını ve metrikleri tek bir API çağrısından alabilmeli.

Inputs:
- `backend/app/api/routes/results.py`

Targets:
- `backend/app/api/routes/results.py`
- `backend/app/api/routes/download.py` (gerekirse)

Implementation Notes:
- Mevcut `/results/{task_id}` endpoint metadata.json'u zaten döndürüyorsa `validation` alanı otomatik akar — sadece response şemasını doğrula
- Mevcut download endpoint segmentation.nii.gz'i sunuyorsa, aynı pattern'la ground_truth.nii.gz için de download yolu ekle: `/results/{task_id}/ground-truth` veya genelleştirilmiş `/results/{task_id}/files/{filename}` (whitelist: segmentation.nii.gz, ground_truth.nii.gz, background.nii.gz)
- Whitelist dışı dosya isteğinde 404
- CORS ve content-type (`application/gzip` veya `application/octet-stream`) doğru ayarlanmalı

Done When:
- [ ] `GET /results/{task_id}` validation alanı mevcutsa response'da görünür
- [ ] `GET /results/{task_id}/files/ground_truth.nii.gz` (veya eşdeğer endpoint) GT dosyasını döner
- [ ] GT yoksa endpoint 404 döner
- [ ] Path traversal koruması: filename whitelist ile

Verification:
- Automated: `pytest tests/backend/api/test_results.py -v`
- Manual: `curl http://localhost:8000/api/results/<task_id>` → JSON'da validation alanı; `curl -O <gt_url>` → dosya iner

---

### TASK-07 — Frontend tip tanımları güncelle
Priority: P1
Model Tier: T1 - Fast
Depends on: TASK-06

Why:
Type-safe veri akışı için backend'in yeni response şemasını TypeScript'e yansıtmak gerek.

Inputs:
- `frontend/src/types/index.ts`

Targets:
- `frontend/src/types/index.ts`

Implementation Notes:
- Yeni tip:
  ```ts
  export interface ValidationMetrics {
    dice: number;
    hd95: number;
  }
  export interface ValidationResult {
    subject_id: string;
    metrics: { WT: ValidationMetrics; TC: ValidationMetrics; ET: ValidationMetrics };
    gt_filename: string;
  }
  ```
- Mevcut `ResultMetadata` (veya benzeri) tipine `validation?: ValidationResult` opsiyonel alan ekle
- Sentinel değerler: HD95 -1 ise UI'da "—" göster (TASK-09'da)

Done When:
- [ ] Tip tanımları eklendi, mevcut tüketici componentler tip kontrolünden geçer (`tsc --noEmit`)

Verification:
- Automated: `cd frontend && npm run build`
- Manual: Yok

---

### TASK-08 — API service GT URL ve metrikler için yardımcı fonksiyonlar
Priority: P1
Model Tier: T1 - Fast
Depends on: TASK-07

Why:
ResultsPage'in GT NIfTI URL'sini ve metrikleri tutarlı şekilde almasını sağlamak.

Inputs:
- `frontend/src/services/api.ts`

Targets:
- `frontend/src/services/api.ts`

Implementation Notes:
- `getGroundTruthUrl(taskId: string): string` yardımcı (aynı segmentation URL pattern'ı ile)
- `getResults(taskId)` fonksiyonu zaten validation alanını dönüyor (TASK-06 sonrası); ekstra fetch yok
- Hata durumlarında validation undefined olabileceğini tüketici tarafa hatırlat (JSDoc)

Done When:
- [ ] `getGroundTruthUrl` fonksiyonu eklendi
- [ ] Mevcut çağrılar bozulmadı

Verification:
- Automated: `cd frontend && npm run build && npm run lint`
- Manual: Yok

---

### TASK-09 — ResultsPage: ikili viewer + metrik kartı
Priority: P0
Model Tier: T3 - Power
Depends on: TASK-07, TASK-08

Why:
Bu taskın kalbi: kullanıcının doğrulama deneyimi. İki viewer'ın senkronize çalışması (kamera/slice paylaşımı opsiyonel ama fayda sağlar) ve metriklerin okunabilir gösterimi UX kalitesini belirler.

Inputs:
- `frontend/src/pages/ResultsPage.tsx`
- `frontend/src/components/SegmentationViewer/SegmentationViewer.tsx`
- TASK-07'deki tipler

Targets:
- `frontend/src/pages/ResultsPage.tsx`
- `frontend/src/pages/ResultsPage.module.css`
- Yeni: `frontend/src/components/ValidationMetricsCard/ValidationMetricsCard.tsx`
- Yeni: `frontend/src/components/ValidationMetricsCard/ValidationMetricsCard.module.css`

Implementation Notes:
- Layout: viewport ≥ 1024px'te iki viewer yan yana (`grid-template-columns: 1fr 1fr`); altta metrik kartı tam genişlik. Mobilde tek sütun, viewer'lar üst üste.
- Viewer üstüne küçük başlık etiketi: "Model Çıktısı" / "Uzman Segmentasyonu (Ground Truth)"
- `validation` alanı yoksa GT viewer ve metrik kartı render edilmez; yerine küçük bir bilgi bandı: "Bu dosya BraTS dataset pattern'ına uymadığı için referans karşılaştırma yapılamadı."
- ValidationMetricsCard:
  - 3 sütunlu küçük tablo: WT | TC | ET satırlar olarak Dice ve HD95
  - Dice değerleri 3 ondalık (`0.938`); HD95 1 ondalık (`4.2 mm` — spacing dahil edildiyse) ya da `4.2 vox`
  - HD95 == -1 ise "—" göster (boş mask edge case'i)
  - Dice renk kodu: ≥0.85 yeşil, 0.70–0.85 sarı, <0.70 kırmızı (color-blind safe palette; renk + ikon birlikte)
  - Açıklama tooltip'i: "WT: tüm tümör, TC: tumor core, ET: enhancing tumor"
- SegmentationViewer'ı GT için yeniden kullan; props olarak farklı NIfTI URL'si geçir (TASK-08)
- İki viewer'ın kamera/slice senkronizasyonu opsiyonel — ilk sürüm için bağımsız çalışsın, scope kontrolünü kaybetmemek için
- Yükleme/hata durumları: GT viewer kendi loading state'ine sahip; bir viewer yüklenirken diğeri çalışmaya devam eder

Done When:
- [ ] BraTS örneği yüklendiğinde sonuç sayfasında iki 3D viewer ve metrik kartı görünür
- [ ] Pattern dışı dosyada sadece tek viewer ve bilgi bandı görünür
- [ ] Metrik değerleri backend'den gelen değerlerle birebir eşleşir (dev tools'tan doğrulanır)
- [ ] Viewer'lar bağımsız çalışır, biri hata verirse diğeri etkilenmez

Verification:
- Automated: `cd frontend && npm run build && npm run lint`
- Manual:
  1. Backend ve frontend dev server'ları başlat
  2. `data/raw/BraTS2021/BraTS2021_00000/` altından 4 modaliteyi yükle, segmentasyonu çalıştır
  3. Sonuç sayfasında iki viewer ve metrik kartını gözle
  4. Farklı isimle yeniden adlandırılmış bir dosyayla tekrar dene → tek viewer + bilgi bandı

---

### TASK-10 — Uçtan uca smoke test ve dokümantasyon
Priority: P1
Model Tier: T2 - Balanced
Depends on: TASK-09

Why:
Özelliği bir testle dondurmak ve nasıl doğrulayacağını projeye yazmak ileride regresyonu yakalamayı kolaylaştırır.

Inputs:
- Çalışan tüm pipeline (TASK-01 → TASK-09)

Targets:
- Yeni: `tests/integration/test_validation_e2e.py`
- `README.md` veya `docs/` altında küçük bir not

Implementation Notes:
- Integration test (frontend hariç):
  1. FastAPI test client ile upload → segment → results akışı çalıştır
  2. BraTS2021_00000 dosyalarını gerçek raw klasörden oku, mock upload
  3. Task tamamlandığında metadata.json'da `validation.metrics.WT.dice >= 0.85` doğrula
  4. GT download endpoint'inin 200 döndüğünü doğrula
- Pattern dışı isimle ikinci senaryo: validation alanının `None`/yok olduğunu doğrula
- README'de "Segmentasyon Doğrulama" bölümü:
  - BraTS pattern gereksinimi
  - Hangi metriklerin hesaplandığı (formüllerle birlikte kısa)
  - Klinik kullanım için olmadığı uyarısı
- Test full inference yapacağı için CPU'da ~dakikalar sürebilir; pytest marker `@pytest.mark.slow` ekle, CI default skip

Done When:
- [ ] Integration test geçer (lokal `pytest -m slow`)
- [ ] README'de doğrulama akışı 5–10 satırlık bölümle anlatıldı

Verification:
- Automated: `pytest tests/integration/test_validation_e2e.py -m slow -v`
- Manual: README'yi gözden geçir

---

## Open Questions
- Spacing için affine'den voxel boyutu çıkarılıyor mu, yoksa BraTS sabit (1mm³) varsayılacak mı? (HD95'in birimini etkiler — mm vs voxel.) → Affine'den çıkarmak doğru olan, BraTS zaten 1mm³ ama generalize için gerekli.
- `medpy` paketini `requirements.txt`'e ekleyelim mi yoksa scipy-tabanlı manuel HD95 implementasyonu yeterli mi? → Manuel implementasyon dependency yükünü azaltır, fakat doğrulanmış bir referansla sayısal olarak karşılaştırılması (TASK-04'te) önemli.
- Iki viewer'ın kamera/slice senkronizasyonu ilk sürümde olmalı mı? Bu plan opsiyonel bırakıyor — eklemek istenirse ek bir TASK-11 olarak tasarlanabilir.
- Production'da raw BraTS klasörü container içinde mevcut olacak mı (Docker volume), yoksa sadece local dev için mi geçerli? Deployment stratejisi netleşmeli.
