from neuralforge.tesseract import TesseractRouter

router = TesseractRouter()
packet = router.route(
    {
        "intent": 0.91,
        "evidence": 0.76,
        "authority": 0.12,
        "context": 0.88,
    },
    mutation_requested=True,
)
print(packet)
