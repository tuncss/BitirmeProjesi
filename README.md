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
