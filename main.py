from spiking_manager import SNNModelManager
from spiking_resnet import SpikingResNet
from spiking_vgg import SpikingVGG

def main():
    # Instantiate the model manager
    manager = SNNModelManager()

    resnet_path = "/path/to/resnet"
    vgg_path = "/path/to/vgg"

    # Instantiate models
    resnet_model = SpikingResNet(model_name="SpikingResNet", model_path=resnet_path)
    vgg_model = SpikingVGG(model_name="SpikingVGG", model_path=vgg_path)

    # Add models to the manager
    manager.add_model(resnet_model)
    manager.add_model(vgg_model)

    # Sample input data for demonstration
    input_data = "sample_input"  # Replace with actual input data

    # Run inference with each model
    print("Running SpikingResNet:")
    resnet_result = manager.run_model("SpikingResNet", input_data)
    print(f"SpikingResNet Result: {resnet_result}")

    print("\nRunning SpikingVGG:")
    vgg_result = manager.run_model("SpikingVGG", input_data)
    print(f"SpikingVGG Result: {vgg_result}")

if __name__ == "__main__":
    main()
