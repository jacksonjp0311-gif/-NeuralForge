from neuralforge.tesseract.train import train_tpn_synthetic

report = train_tpn_synthetic(n=384, epochs=2, batch_size=32, d_model=48)
print(report["final"])
