# Brain Tumor Segmentation

3D U-Net ve 3D Attention U-Net mimarileriyle beyin MR goruntulerinde otomatik tumor segmentasyonu yapan web tabanli bitirme projesi.

## Proje Ozeti

Bu proje, BraTS 2021 veri seti uzerinde egitilen PyTorch modellerini FastAPI backend ve React frontend ile birlestirerek klinik karar destek odakli bir segmentasyon sistemi gelistirmeyi hedefler.

## Teknoloji Yigini

- PyTorch 2.x
- FastAPI
- React 18 + Vite + TypeScript
- Niivue
- Celery + Redis
- Docker Compose

## Kurulum

Kurulum adimlari proje altyapisi tamamlandikca detaylandirilacaktir.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Gelistirme Durumu

Proje Faz 1 altyapi calismalariyla baslatilmistir.

## Segmentasyon Dogrulama

BraTS dogrulama akisi, yuklenen dosya adlari `BraTS2021_XXXXX_<modality>.nii.gz` pattern'ina uydugunda `data/raw/BraTS2021/<subject_id>/<subject_id>_seg.nii.gz` referans maskesini otomatik bulur. Backend inference sonrasinda WT `{1,2,4}`, TC `{1,4}` ve ET `{4}` bolgeleri icin Dice ve HD95 hesaplar, `metadata.json` icindeki `validation` alanina yazar ve `ground_truth.nii.gz` dosyasini result klasorune kopyalar. Pattern disi yuklemelerde dogrulama alani olusmaz; segmentation sonucu yine normal sunulur. HD95 affine'den turetilen voxel spacing ile hesaplanir; tek taraf bos maskelerde UI icin `-1.0` sentinel degeri kullanilir. Bu metrikler arastirma/prototip amaclidir, klinik karar veya regulatory onay iddiasi tasimaz.
