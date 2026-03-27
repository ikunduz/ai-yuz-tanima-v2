# AI Yüz Tanıma Prototipi

## Amaç

Bu çalışma, bilim müzesi için planlanan interaktif "AI mirror" exhibit'inin ilk yerel prototipini MacBook Pro üzerinde, dahili webcam kullanarak geliştirmek içindir. İlk sürüm tamamen lokal çalışacaktır; internet bağlantısı ve harici AI API'si zorunlu değildir.

## İlk Prototip Hedefi

- MacBook dahili kamerasından canlı görüntü almak
- Yüzü gerçek zamanlı tespit etmek
- Yüz landmark noktalarını çıkarmak
- Duygu sınıflandırması yerine güvenli ve açıklanabilir yüz sinyalleri göstermek
- Ekran üzerine canlı overlay çizmek
- Mimariyi daha sonra Raspberry Pi, Jetson veya mini PC'ye taşınabilir tasarlamak

## Neden Local-First

- Gecikme düşer
- İnternet kesintisinden etkilenmez
- Müze kurulumunda operasyon kolaylaşır
- KVKK ve mahremiyet açısından daha güvenli yaklaşım sağlar
- Bulut API maliyeti oluşmaz

## Prototipte Gösterilecek Veriler

İlk sürümde doğrudan yaş/cinsiyet üretmek yerine aşağıdaki açıklanabilir metrikler gösterilecektir:

- yüz bulundu / bulunamadı
- takip güveni
- ağız açıklığı
- gülümseme benzeri oran
- göz açıklığı
- kaş yüksekliği benzeri oran
- baş eğimi / yatıklık
- fps

Bu yaklaşım sergi anlatısı için daha sağlamdır. İstenirse ikinci aşamada ayrı modellerle yaş aralığı veya ifade kategorileri eklenebilir.

## Yazılım Mimarisi

### 1. Camera Layer

Sorumluluk:

- kamerayı açmak
- frame almak
- çözünürlük ayarlamak
- farklı kamera kaynaklarına taşınabilir olmak

İlk kaynak:

- MacBook internal webcam

Sonraki kaynaklar:

- USB webcam
- Raspberry Pi Camera
- Intel RealSense

### 2. Inference Layer

Sorumluluk:

- yüz bulmak
- yüz landmark'larını çıkarmak
- ham ölçümleri hesaplamak

İlk sürüm:

- MediaPipe Face Landmarker Tasks tabanlı çözüm

Sonraki sürüm:

- MediaPipe Tasks veya ONNX tabanlı modüler modeller

### 3. Metrics Layer

Sorumluluk:

- landmark'lardan açıklanabilir oranlar üretmek
- metrikleri normalize etmek
- zaman içinde yumuşatmak

Örnek:

- mouth_open
- smile_proxy
- eye_open_left / eye_open_right
- brow_raise
- head_tilt

### 4. Presentation Layer

Sorumluluk:

- kamera görüntüsü üzerine kutu, landmark ve bilgi paneli çizmek
- tam ekran prototip görünümü üretmek
- ileride kiosk ekranına uygun bir UI'ye dönüşmek

İlk sürüm:

- OpenCV penceresi üzerinden overlay

Sonraki sürüm:

- web tabanlı veya kiosk arayüzü

### 5. Config Layer

Sorumluluk:

- çözünürlük
- kamera indeksi
- maksimum yüz sayısı
- confidence eşikleri
- smoothing gücü
- aynalama seçeneği

Bu parametreler koddan ayrılmalıdır; cihaz değiştiğinde aynı uygulama sadece config ile uyarlanabilmelidir.

## Veri Akışı

1. Kamera frame üretir.
2. Frame RGB'ye çevrilir.
3. Inference katmanı yüzü ve landmark'ları çıkarır.
4. Metrics katmanı oranları hesaplar.
5. Smoothing katmanı anlık dalgalanmayı azaltır.
6. Overlay katmanı görselleştirir.
7. Kullanıcı `q` veya `ESC` ile çıkar.

## Prototip Klasör Yapısı

```text
ai-yuz-tanima/
  requirements.txt
  README.md
  src/
    main.py
    config.py
    camera.py
    analyzer.py
    overlay.py
```

## Uygulama Kararları

### Bu aşamada bilgisayar şart mı?

Hayır. Nihai ürün için şart değil.

Fakat ilk prototip için MacBook en doğru başlangıç ortamıdır:

- elindeki donanım hazır
- hızlı iterasyon yapılır
- webcam, ekran ve debug ortamı tek cihazda bulunur
- algoritma kalitesi donanımdan önce doğrulanır

### Raspberry Pi yeterli olur mu?

Olabilir ama iki seviyede düşünülmelidir:

- prototip / basit demo: Raspberry Pi 5 bazen yeterli olabilir
- kalıcı müze kurulumu: Pi 5 + AI hızlandırıcı veya daha güçlü edge cihaz daha güvenlidir

Bu yüzden mimari taşınabilir tasarlanacaktır; ilk doğrulama Mac üzerinde yapılacaktır.

### İnternet veya AI API gerekecek mi?

İlk sürüm için hayır.

Zorunlu olmayan ek kullanım örnekleri:

- uzaktan log gönderme
- exhibit analitikleri
- bulut tabanlı anlatıcı / sohbet ajanı
- merkezi içerik güncelleme

## Geliştirme Aşamaları

### Aşama 1: Çalışan Yerel Prototype

- webcam bağlantısı
- yüz landmark takibi
- temel overlay
- fps göstergesi

Başarı kriteri:

- tek kişi karşısında akıcı görüntü
- 10 FPS ve üzeri deneyim

### Aşama 2: Exhibit Davranışı

- en baskın yüzü seçme
- kararlı skor üretme
- yüz kaybolduğunda bekleme moduna dönme

### Aşama 3: Kiosklaştırma

- tam ekran
- otomatik başlama
- hata sonrası yeniden açılma
- sade ve büyük tipografili UI

### Aşama 4: Donanım Geçişi

- USB webcam ile test
- farklı çözünürlüklerde performans testi
- hedef cihaz için inference backend seçimi

## Teknik Riskler

- düşük ışıkta landmark kalitesi düşebilir
- çok kişi varsa yanlış yüz seçimi olabilir
- çocuk yüzlerinde oranlar farklı davranabilir
- emotion etiketi doğrudan verilirse yanlış yorum riski oluşur

## Tasarım İlkeleri

- Local-first
- Açıklanabilir metrikler
- Modüler kod
- Donanımdan bağımsız camera abstraction
- Müze ortamına uygun sade arayüz

## Bu Dosyanın Sonrası İçin İlk İşler

1. `ai-yuz-tanima/` altında Python iskeletini oluştur
2. webcam yakalama kodunu ekle
3. landmark çıkarımını bağla
4. overlay panelini ekle
5. MacBook üzerinde ilk canlı testi yap
6. performans ve ışık koşullarını not al

Not:

- Face Landmarker model dosyası ilk çalıştırmada otomatik indirilecektir ve sonra lokalden kullanılacaktır.

## Mevcut Karar

İlk implementasyon hedefi:

- cihaz: MacBook Pro
- kamera: internal webcam
- ağ: kapalı olabilir
- çalışma modu: tamamen local
- dil: Python
- görüntüleme: OpenCV window

## Onayli Kalibrasyon Notu

Tarih:

- 26 Mart 2026

Bu tarihte `src/analyzer.py` icindeki ifade kalibrasyonu birlikte test edilerek onaylandi.
Bu kalibrasyon, kullanicidan yeni bir istek gelmedikce referans durum olarak korunmalidir.

Onaylanan davranis:

- `Mutlu` agirlikla agiz sekli, gulumseme ve yanak sinyalinden beslenir.
- `Saskin` agirlikla acik goz + yuvarlak acik agiz kombinasyonundan beslenir.
- `Kizgin` agirlikla kas catma, kaslarin asagi inmesi ve iki kasin birbirine yaklasmasindan beslenir.
- `Kizgin` sinyali, `Mutlu` ve `Saskin` sinyallerinden ayrismis durumda tutulmalidir.
- Nötr yuzde `Kizgin` gereksiz yere tavana vurmamali; belirgin catik kas ifadesinde ise hizli yukselmelidir.

Uygulama kurali:

- Bu kalibrasyon ancak kullanici acikca isterse tekrar degistirilir.
