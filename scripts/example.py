import os
from typing import Dict, Tuple, List, Optional

import coremltools as ct
import onnxruntime
import torch
import torch.nn.functional as F
import json
from PIL import Image
from uform.models import PreProcessor, VLM
from functools import partial
import time

# export TOKENIZERS_PARALLELISM=true

def preprocess_data(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.method == "onnx":
            result = {k: v.cpu().numpy() for k, v in result[0].items()}, {
                self.input_image_name: result[1].cpu().numpy()
            }
        elif self.method == "coreml":
            input_ids = result[0]["input_ids"].type(torch.int32).cpu().numpy()
            attention_mask = result[0]["attention_mask"].type(torch.int32).cpu().numpy()

            result = {"input_ids": input_ids, "attention_mask": attention_mask}, {
                self.input_image_name: result[1].cpu().numpy()
            }
        return result

    return wrapper

def get_local_model(model_name: str, token: Optional[str] = None) -> VLM:
    config_path = f"{model_name}/torch_config.json"
    state = torch.load(f"{model_name}/torch_weight.pt")

    tokenizer_path = f"{model_name}/tokenizer.json"

    with open(config_path, "r") as f:
        model = VLM(json.load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()


class MyModel:
    def __init__(self, method: str, model_fpath: str) -> None:
        self.method = method
        self.model_fpath = model_fpath
        max_position_embeddings = 50
        if method == "torch":
            self.model = get_local_model(model_fpath)
            self.image_model = partial(self.model.encode_image, return_features=True)
            self.text_model = partial(self.model.encode_text, return_features=True)
        elif method == "onnx":
            fname = "multilingual.{}-encoder.onnx"
            image_ort_session = onnxruntime.InferenceSession(
                os.path.join(model_fpath, fname.format("image")), providers=["CPUExecutionProvider"]
            )
            text_ort_session = onnxruntime.InferenceSession(
                os.path.join(model_fpath, fname.format("text")), providers=["CPUExecutionProvider"]
            )
            input_ids = text_ort_session.get_inputs()[0]
            max_position_embeddings = input_ids.shape[-1]

            def predict_func(ort_session, data):
                out = ort_session.run(None, data)
                return torch.tensor(out[0]), torch.tensor(out[1])

            self.image_model = partial(predict_func, image_ort_session)
            self.text_model = partial(predict_func, text_ort_session)

            input_image = image_ort_session.get_inputs()[0]
            self.input_image_name = input_image.name
        elif method == "coreml":
            fname = "multilingual-v2.{}-encoder.mlpackage"
            image_mlmodel = ct.models.MLModel(os.path.join(model_fpath, fname.format("image")))
            text_mlmodel = ct.models.MLModel(os.path.join(model_fpath, fname.format("text")))

            def predict_func(model, data):
                out = model.predict(data)
                return torch.tensor(out["features"]), torch.tensor(out["embeddings"])

            self.image_model = partial(predict_func, image_mlmodel)
            self.text_model = partial(predict_func, text_mlmodel)

            input_image = image_mlmodel.input_description._fd_spec[0]
            input_text_lst = text_mlmodel.input_description._fd_spec
            self.input_image_name = input_image.name

        self.preprocess = PreProcessor(
            os.path.join(self.model_fpath, "tokenizer.json"), max_position_embeddings, 1, 224
        )

    @preprocess_data
    def preprocess_text_image(self, text, image) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        image_data = self.preprocess.preprocess_image(image)
        text_data = self.preprocess.preprocess_text(text)
        return text_data, image_data

    def forward(self, text_data, image_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features, image_embedding = self.image_model(image_data)
        text_features, text_embedding = self.text_model(text_data)
        return image_features, image_embedding, text_features, text_embedding

    def __call__(self, text: str, image) -> float:
        text_data, image_data = self.preprocess_text_image(text, image)
        image_features, image_embedding = self.image_model(image_data)
        text_features, text_embedding = self.text_model(text_data)

        similarity = F.cosine_similarity(image_embedding, text_embedding)
        if self.method == "torch":
            joint_embedding = self.model.encode_multimodal(
                image_features=image_features,
                text_features=text_features,
                attention_mask=text_data["attention_mask"],
            )
            score = self.model.get_matching_scores(joint_embedding)
            print("torch score", score)
        return similarity

if __name__ == "__main__":
    text = 'a small red panda in a zoo'
    image = Image.open('red_panda.jpg')
    model_fpath = ...

    for method in ["torch", "onnx", "coreml"]:
        model = MyModel(method, model_fpath)
        text_data, image_data = model.preprocess_text_image(text, image)
        model.forward(text_data, image_data) # just for warm-up
        loop_cnt = 10
        s1 = time.time()
        for _ in range(loop_cnt):
            model.forward(text_data, image_data)
        print(method, time.time() - s1)