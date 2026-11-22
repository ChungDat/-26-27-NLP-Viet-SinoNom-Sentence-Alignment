import unittest
import numpy as np

from src.preprocess import (
    COLOR_BINARY,
    COLOR_GRAYSCALE,
    COLOR_RGB,
    BaseTransformation,
    BinaryMorphologyTransform,
    ConvertColorTransform,
    FilterTransform,
    GrayscaleMorphologyTransform,
    PreprocessPipeline,
    ReverseColorTransform,
    binary_morphology,
    convert_color,
    grayscale_morphology,
    remove_artifact,
    reverse_color,
)


class TestPreprocessPipeline(unittest.TestCase):
    def test_empty_pipeline_returns_input_image_directly(self):
        pipeline = PreprocessPipeline()
        # Test with float array
        img_float = np.array([[0.1, 0.5], [0.8, 1.0]], dtype=np.float32)
        out_float = pipeline.apply(img_float)
        self.assertIs(out_float, img_float)

        # Test with uint8 array
        img_uint8 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        out_uint8 = pipeline.apply(img_uint8)
        self.assertIs(out_uint8, img_uint8)

    def test_available_transformations_registry(self):
        transforms = PreprocessPipeline.get_available_transformations()
        expected = {
            "convert_color",
            "reverse_color",
            "filter",
            "binary_morphology",
            "grayscale_morphology",
        }
        self.assertTrue(expected.issubset(set(transforms.keys())))
        for name, cls in transforms.items():
            self.assertTrue(issubclass(cls, BaseTransformation))
            self.assertTrue(hasattr(cls, "label") and cls.label)
            self.assertTrue(hasattr(cls, "param_specs"))

    def test_pipeline_add_move_remove_step(self):
        pipeline = PreprocessPipeline()
        self.assertEqual(len(pipeline.steps), 0)

        pipeline.add_step("convert_color", {"mode": COLOR_GRAYSCALE})
        pipeline.add_step("reverse_color")
        pipeline.add_step("filter", {"kernel_size": 5})
        self.assertEqual(len(pipeline.steps), 3)
        self.assertEqual(pipeline.steps[0].name, "convert_color")
        self.assertEqual(pipeline.steps[1].name, "reverse_color")
        self.assertEqual(pipeline.steps[2].name, "filter")

        # Move step 2 to index 0
        pipeline.move_step(2, 0)
        self.assertEqual(pipeline.steps[0].name, "filter")
        self.assertEqual(pipeline.steps[1].name, "convert_color")
        self.assertEqual(pipeline.steps[2].name, "reverse_color")

        # Update step params
        pipeline.update_step(0, {"kernel_size": 7})
        self.assertEqual(pipeline.steps[0].params["kernel_size"], 7)

        # Remove step
        pipeline.remove_step(1)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0].name, "filter")
        self.assertEqual(pipeline.steps[1].name, "reverse_color")

    def test_pipeline_serialization_and_deserialization(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step("convert_color", {"mode": COLOR_BINARY})
        pipeline.add_step("binary_morphology", {"operation": "open", "kernel_size": 3})

        serialized = pipeline.to_dict()
        self.assertEqual(len(serialized), 2)
        self.assertEqual(serialized[0]["type"], "convert_color")
        self.assertEqual(serialized[0]["params"]["mode"], COLOR_BINARY)

        # From dict format v1
        settings = {"version": 1, "steps": serialized}
        pipeline2 = PreprocessPipeline.from_dict(settings)
        self.assertEqual(len(pipeline2.steps), 2)
        self.assertEqual(pipeline2.steps[0].name, "convert_color")
        self.assertEqual(pipeline2.steps[1].name, "binary_morphology")

        # Dict compatibility on step objects
        step = pipeline2.steps[0]
        self.assertEqual(step["type"], "convert_color")
        self.assertEqual(step.get("type"), "convert_color")
        self.assertEqual(step["params"]["mode"], COLOR_BINARY)

    def test_pipeline_apply(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step("convert_color", {"mode": COLOR_GRAYSCALE})
        pipeline.add_step("reverse_color")

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        processed = pipeline.apply(img)
        self.assertEqual(processed.ndim, 2)
        self.assertEqual(processed.shape, (10, 10))
        # Since black (0) was inverted, it should be white (255)
        self.assertTrue(np.all(processed == 255))

    def test_alias_remove_artifact(self):
        pipeline = PreprocessPipeline()
        pipeline.add_step("remove_artifact", {"kernel_size": 3})
        self.assertEqual(pipeline.steps[0].name, "filter")


if __name__ == "__main__":
    unittest.main()
