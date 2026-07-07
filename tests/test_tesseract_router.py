from neuralforge.tesseract import AxisScores, TesseractRouter, build_route_state


def test_authority_missing_mutation_routes_shadow():
    router = TesseractRouter()
    packet = router.route(
        {"intent": 0.95, "evidence": 0.90, "authority": 0.05, "context": 0.92},
        mutation_requested=True,
    )
    assert packet["vertex"] == "1101"
    assert packet["route"] == "shadow"
    assert packet["missing_axes"] == ["authority"]


def test_full_ready_routes_engage():
    state = build_route_state(AxisScores(0.95, 0.90, 0.91, 0.88), mutation_requested=False)
    assert state.vertex == "1111"
    assert state.route == "engage"
    assert state.coherence > 0.70


def test_selected_experts_include_current_vertex():
    router = TesseractRouter()
    packet = router.route({"intent": 0.1, "evidence": 0.1, "authority": 0.1, "context": 0.1})
    experts = packet["selected_experts"]
    assert experts[0]["vertex"] == "0000"
    assert len(experts) >= 2
