import os
import typing
import time
import onnxruntime as ort
import cv2
import numpy as np
import yaml

from collections import deque
from itertools import groupby


class FpsWrapper:
    """ Decorator to calculate the frames per second of a function
    """
    def __init__(self, func: typing.Callable):
        self.func = func
        self.fps_list = deque([], maxlen=100)

    def __call__(self, *args, **kwargs):
        start = time.time()
        results = self.func(self.instance, *args, **kwargs)
        self.fps_list.append(1 / (time.time() - start))
        self.instance.fps = np.mean(self.fps_list)
        return results

    def __get__(self, instance, owner):
        self.instance = instance
        return self.__call__.__get__(instance, owner)

class BaseModelConfigs:
    def __init__(self):
        self.model_path = None

    def serialize(self):
        class_attributes = {key: value
                            for (key, value)
                            in type(self).__dict__.items()
                            if key not in ['__module__', '__init__', '__doc__', '__annotations__']}
        instance_attributes = self.__dict__

        # first init with class attributes then apply instance attributes overwriting any existing duplicate attributes
        all_attributes = class_attributes.copy()
        all_attributes.update(instance_attributes)

        return all_attributes

    def save(self, name: str = "configs.yaml"):
        if self.model_path is None:
            raise Exception("Model path is not specified")

        # create directory if not exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, name), "w") as f:
            yaml.dump(self.serialize(), f)

    @staticmethod
    def load(configs_path: str):
        with open(configs_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        config = BaseModelConfigs()
        for key, value in configs.items():
            setattr(config, key, value)

        return config

class OnnxInferenceModel:
    """ Base class for all inference models that use onnxruntime 

    Attributes:
        model_path (str, optional): Path to the model folder. Defaults to "".
        force_cpu (bool, optional): Force the model to run on CPU or GPU. Defaults to GPU.
        default_model_name (str, optional): Default model name. Defaults to "model.onnx".
    """
    def __init__(
        self, 
        model_path: str = "",
        force_cpu: bool = False,
        default_model_name: str = "model.onnx",
        *args, **kwargs
        ):
        self.model_path = model_path.replace("\\", "/")
        self.force_cpu = force_cpu
        self.default_model_name = default_model_name

        # check if model path is a directory with os path
        if os.path.isdir(self.model_path):
            self.model_path = os.path.join(self.model_path, self.default_model_name)

        if not os.path.exists(self.model_path):
            raise Exception(f"Model path ({self.model_path}) does not exist")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" and not force_cpu else ["CPUExecutionProvider"]

        self.model = ort.InferenceSession(self.model_path, providers=providers)

        self.metadata = {}
        if self.model.get_modelmeta().custom_metadata_map:
            # add metadata to self object
            for key, value in self.model.get_modelmeta().custom_metadata_map.items():
                try:
                    new_value = eval(value) # in case the value is a list or dict
                except:
                    new_value = value
                self.metadata[key] = new_value
                
        # Update providers priority to only CPUExecutionProvider
        if self.force_cpu:
            self.model.set_providers(["CPUExecutionProvider"])

        self.input_shapes = [meta.shape for meta in self.model.get_inputs()]
        self.input_names = [meta.name for meta in self.model._inputs_meta]
        self.output_names = [meta.name for meta in self.model._outputs_meta]

    def predict(self, data: np.ndarray, *args, **kwargs):
        raise NotImplementedError

    @FpsWrapper
    def __call__(self, data: np.ndarray):
        results = self.predict(data)
        return results


def ctc_decoder(predictions: np.ndarray, chars: typing.Union[str, list]) -> typing.List[str]:
    """ CTC greedy decoder for predictions

    Args:
        predictions (np.ndarray): predictions from model
        chars (typing.Union[str, list]): list of characters

    Returns:
        typing.List[str]: list of words
    """
    # use argmax to find the index of the highest probability
    argmax_preds = np.argmax(predictions, axis=-1)

    # use groupby to find continuous same indexes
    grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

    # convert indexes to chars
    texts = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]

    return texts

def get_cer(
    preds: typing.Union[str, typing.List[str]],
    target: typing.Union[str, typing.List[str]],
    ) -> float:
    """ Update the cer score with the current set of references and predictions.

    Args:
        preds (typing.Union[str, typing.List[str]]): list of predicted sentences
        target (typing.Union[str, typing.List[str]]): list of target words

    Returns:
        Character error rate score
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer


def edit_distance(prediction_tokens: typing.List[str], reference_tokens: typing.List[str]) -> int:
    """ Standard dynamic programming algorithm to compute the Levenshtein Edit Distance Algorithm

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    # Initialize a matrix to store the edit distances
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]

    # Fill the first row and column with the number of insertions needed
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j

    # Iterate through the prediction and reference tokens
    for i, p_tok in enumerate(prediction_tokens):
        for j, r_tok in enumerate(reference_tokens):
            # If the tokens are the same, the edit distance is the same as the previous entry
            if p_tok == r_tok:
                dp[i+1][j+1] = dp[i][j]
            # If the tokens are different, the edit distance is the minimum of the previous entries plus 1
            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1

    # Return the final entry in the matrix as the edit distance     
    return dp[-1][-1]

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    configs = BaseModelConfigs.load("Models/1_image_to_word/202211270035/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/1_image_to_word/202211270035/val.csv").dropna().values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df[:20]):
        image = cv2.imread(image_path.replace("\\", "/"))

        try:
            prediction_text = model.predict(image)

            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            # resize image by 3 times for visualization
            # image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3))
            # cv2.imshow(prediction_text, image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        except:
            continue
        
        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")