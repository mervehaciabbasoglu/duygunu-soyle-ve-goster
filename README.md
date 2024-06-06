# duygunu soyle ve goster
Duygunu Söyle ve Göster (İngilizce) | Merve Hacıabbasoğlu

Bu uygulama, gerçek zamanlı olarak yüz ifadelerini ve sesleri algılayarak duygusal durumumuzu doğruluk oranı ile ifade eder. Bu README dosyası, projeyle ilgili detaylı bilgileri içermektedir.
İçindekiler
1.	Proje Dosyaları ve Klasörler
2.	Kullanım Talimatları
3.	Gereksinimler
4.	Projenin İndirilmesi ve Kurulumu
Proje Dosyaları ve Klasörler
•	model_ses.py: Ses verilerinden özellikler çıkararak derin öğrenme kullanımıyla bir duygusal durum tanıma modeli oluşturur, modeli eğitir ve diske kaydeder.
•	model_yuz.py: Eğitilmiş yüz tanıma modelini kullanarak, kameradan alınan görüntülerde yüzleri algılar ve bu yüzlerin duygusal durumlarını tahmin eder.
•	realtime_main.py: Kameradan gerçek zamanlı olarak alınan görüntülerde yüz ifadelerini tanır ve bu yüz ifadelerini görsel olarak işaretler.
•	sestanima_main.py: Kullanıcı arayüzü sağlar, ses kaydı yapmanıza, kaydedilen ses dosyasını oynatmanıza, analiz etmenize ve model aracılığıyla duygusal durumu tahmin etmenize olanak tanır.
Kullanım Talimatları
Modelin Eğitimi
1.	Ses Tanıma Modeli:
o	model_ses.py dosyasını çalıştırarak ses tanıma modelini eğitin ve sestanima_modeli.h5 dosyasına kaydedin.
2.	Yüz Tanıma Modeli:
o	model_yuz.py dosyasını çalıştırarak yüz tanıma modelini eğitin ve yuz_tanima_modeli.h5 dosyasına kaydedin.
Duygusal Durum Tahmini
1.	Ses Tanıma:
o	sestanima_main.py dosyasını çalıştırarak kullanıcı arayüzünü açın.
o	Ses kaydı yapın veya bir ses dosyası yükleyin ve tahmin sonuçlarını görün.
2.	Yüz Tanıma:
o	realtime_main.py dosyasını çalıştırarak kamerayı açın.
o	Anlık olarak yüz ifadenizi kameraya yapın ve algıladığı yüz ifadesini yüzdelik olarak görün.
Gereksinimler
•	Python 3.x
•	TensorFlow
•	NumPy
•	Librosa
•	Sounddevice
•	Soundfile
•	Tkinter (GUI için)
Projenin İndirilmesi ve Kurulumu
Proje zip dosyası Google Drive üzerinde mevcuttur. Aşağıdaki linkten projeyi zip halinde indirebilir, zipten çıkardıktan sonra kullanabilirsiniz.
Proje Linki: https://drive.google.com/file/d/12vrdkxu7XhlQXp3s4pY1DVG9cNxBgGKi/view?usp=sharing

