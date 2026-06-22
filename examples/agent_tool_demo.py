import neuralforge as nf

def main():
    print("NeuralForge v2.0 - Agent Tool Demo")
    tool = nf.as_tool()
    print(f"Tool: {tool.name}, Actions: {len(tool.get_available_actions())}")
    r = tool.invoke({"action": "create_model", "description": "CNN for CIFAR-10 with < 5M params"})
    print(f"Create: {r['status']}, Params: {r['result']['parameters']:,}")
    r2 = tool.invoke({"action": "full_pipeline", "description": "image classifier for CIFAR-10"})
    print(f"Pipeline: {r2['status']}")
    models = nf.list_models()
    print(f"Registered models: {len(models)}")
    print("Done!")

if __name__ == "__main__": main()
