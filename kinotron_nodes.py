from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import comfy.utils
import time
import random
from openai import OpenAI


class TextList:
    def __init__(self, rnd):
        self.items = []
        self.id = rnd

    def add(self, item):
        """Appends a new item to the list."""
        self.items.append(item)

    def get(self, n):
        return self.items[n]

    def size(self):
        return len(self.items)

    def clear(self):
        self.items = []

#IO
class TextInput:
    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING",{"multiline": True}),
            },
        }

    CATEGORY = "Kinotron/Text"
    FUNCTION = "output"
    RETURN_NAMES = ("text",)
    RETURN_TYPES = ("STRING",)

    def output(self,input, **inputs):
        return (input,)

class ToConsole:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    FUNCTION = "show_text"
    CATEGORY = "Kinotron/Output"

    def show_text(self, text):
        print(text)
        return text

class ToConsoleList:
    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("LIST", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True

    FUNCTION = "show_text"
    CATEGORY = "Kinotron/Output"

    def show_text(self, list):
        tmp = ""
        for i in range(0,list.size()):
            tmp = tmp+"\n"+list.get(i)
        list.clear()
        print(tmp)
        return tmp

class Combine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scene_1": ("STRING", {"multiline": False, "forceInput": True}),
            },
            "optional": {
                "scene_2": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_3": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_4": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_5": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_6": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_7": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_8": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_9": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_10": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_11": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_12": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_13": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_14": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_15": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_16": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_17": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_18": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_19": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_20": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("output_list",)
    OUTPUT_NODE = True

    FUNCTION = "combine"
    CATEGORY = "Kinotron"

    def combine(self, scene_1,scene_2,scene_3,scene_4,scene_5,scene_6,scene_7,scene_8,scene_9,scene_10,
                scene_11,scene_12,scene_13,scene_14,scene_15,scene_16,scene_17,scene_18,scene_19,scene_20, **inputs):
        return (scene_1+"###"+scene_2+"###"+scene_3+"###"+scene_4+"###"+scene_5+"###"+scene_6+"###"+scene_7+"###"+scene_8+"###"+scene_9+"###"+scene_10+"###"+scene_11+"###"+scene_12+"###"+scene_13+"###"+scene_14+"###"+scene_15+"###"+scene_16+"###"+scene_17+"###"+scene_18+"###"+scene_19+"###"+scene_20,)

class EmptyList:
    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)

    FUNCTION = "empty"
    CATEGORY = "Kinotron/List"

    def empty(self,**inputs):
        tmp = TextList(random.randint(1, 99999999))
        return (tmp,)

class Append:
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("LIST", {"multiline": False, "forceInput": True}),
                "scene": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "append"
    CATEGORY = "Kinotron/List"

    def append(self, list,scene, **inputs):
        list.add(scene)
        #print(list.items)
        return (list,)

class Append2:
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("LIST", {"multiline": False, "forceInput": True}),
                "scene_1": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_2": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "append"
    CATEGORY = "Kinotron/List"

    def append(self, list,scene_1,scene_2, **inputs):
        list.add(scene_1)
        list.add(scene_2)
        tmp = list
        #print(list.items)
        return (tmp,)

class Append3:
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("LIST", {"multiline": False, "forceInput": True}),
                "scene_1": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_2": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_3": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "append"
    CATEGORY = "Kinotron/List"

    def append(self, list,scene_1,scene_2,scene_3, **inputs):
        list.add(scene_1)
        list.add(scene_2)
        list.add(scene_3)
        #print(list.items)
        return (list,)

class Append5:
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("LIST", {"multiline": False, "forceInput": True}),
                "scene_1": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_2": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_3": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_4": ("STRING", {"multiline": False, "forceInput": True}),
                "scene_5": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("list",)
    FUNCTION = "append"
    CATEGORY = "Kinotron/List"

    def append(self, list,scene_1,scene_2,scene_3,scene_4,scene_5, **inputs):
        list.add(scene_1)
        list.add(scene_2)
        list.add(scene_3)
        list.add(scene_4)
        list.add(scene_5)
        #print(list.items)
        return (list,)

#BASIC OPENAI LLM GENERATION

class LoadOpenAI:
    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"forceInput": False}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron"
    FUNCTION = "load"
    RETURN_NAMES = ("model",)
    RETURN_TYPES = ("OBJECT",)

    def load(self,api_key, **inputs):
        print("Loaded OpenAI model")
        return (OpenAI(api_key=api_key),)

class Inference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "prompt": ("STRING", {"multiline": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron"
    FUNCTION = "inference"
    RETURN_NAMES = ("output",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,prompt, **inputs):
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)
        
class InferenceTextInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "prompt": ("STRING", {"multiline": True}),
                "text": ("STRING", {"multiline": False, "forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron"
    FUNCTION = "inference"
    RETURN_NAMES = ("output",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,prompt,text, **inputs):
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"Execute the user's prompt on this text: {text}"},
                {"role": "user",
                 "content": prompt}
            ]
        )
        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)

#ARCHETYPES
class ThreeAct:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "synopsis": ("STRING", {"multiline": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Archetypes"
    FUNCTION = "inference"
    RETURN_NAMES = ("Act 1","Act 2","Act 3",)
    RETURN_TYPES = ("STRING","STRING","STRING",)

    def inference(self,model,synopsis, **inputs):
        prompt = f"SYNOPSIS: {synopsis} \n Given this synopsis write a brief description for each of the story's acts, one per line. \n THREE ACTS SUMMARIES:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        acts = response.choices[0].message.content.split("\n")
        acts = [string for string in acts if string]
        print(acts)
        return (acts[0],acts[1],acts[2],)

class Freytag:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "synopsis": ("STRING", {"multiline": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Archetypes"
    FUNCTION = "inference"
    RETURN_NAMES = ("Exposition","Inciting incident","Rising action","Climax","Falling action","Denouement",)
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING",)

    def inference(self,model,synopsis, **inputs):
        prompt = f"SYNOPSIS: {synopsis} \n Given this synopsis write a brief description for each of Freytag's 6 pyramid chapters, one per line. \n FREYTAG'S SUMMARIES:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        acts = response.choices[0].message.content.split("\n")
        acts = [string for string in acts if string]
        print(acts)
        return (acts[0],acts[1],acts[2],acts[3],acts[4],acts[5])

#TREE EXPANSION
class Expand2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"forceInput": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Hierarchical"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene_1","scene_2")
    RETURN_TYPES = ("OBJECT","STRING","STRING",)

    def inference(self,model,scene, **inputs):
        print("Expanding into 2 scenes...")
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a story machine that takes as input from the user a story scene summary and must produce from it 2 short separate scene summaries that deepen the narrative. Write your output in 1 line, separated the scenes by ###, DO NOT NUMBER THE SCENES. Do not write anything elese."},
                {"role": "user",
                 "content": scene}
            ]
        )
        scenes = response.choices[0].message.content.split("###")
        for scene in scenes:
            scene.replace("\n","")
        return (model,scenes[0],scenes[1],)

class Expand3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"forceInput": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Hierarchical"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene_1","scene_2","scene_3")
    RETURN_TYPES = ("OBJECT","STRING","STRING","STRING",)

    def inference(self, model, scene, **inputs):
        print("Expanding into 2 scenes...")
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a story machine that takes as input from the user a story scene summary and must produce from it 3 short separate scene summaries that deepen the narrative. Write your output in 1 line, separated the scenes by ###, DO NOT NUMBER THE SCENES. Do not write anything elese."},
                {"role": "user",
                 "content": scene}
            ]
        )
        scenes = response.choices[0].message.content.split("###")
        for scene in scenes:
            scene.replace("\n","")
        return (model, scenes[0], scenes[1], scenes[2])

class Interpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene_A": ("STRING", {"multiline": False, "forceInput":True}),
                "scene_B": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Hierarchical"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","interpolated",)
    RETURN_TYPES = ("OBJECT","STRING",)

    def inference(self,model,scene_A,scene_B, **inputs):
        prompt = f"SCENE A SUMMARY: {scene_A} \n SCENE B SUMMARY: {scene_B} \n Generate the summary for a possible scene inbetween A and B. \n INBETWEEN SCENE SUMMARY:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        return (model,response.choices[0].message.content,)

#SEQUENTIAL
class NextScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": False, "forceInput":True}),
                "list": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Sequential"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene","list")
    RETURN_TYPES = ("OBJECT","STRING","STRING")

    def inference(self,model,scene,list, **inputs):
        prompt = f"CURRENT SCENE SUMMARY: {scene} \n Generate the next scene summary in the plotline. \n NEXT SCENE SUMMARY:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        return (model,response.choices[0].message.content,list+"###\n"+response.choices[0].message.content)

class FirstScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "input": ("STRING", {"forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Sequential"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene","list")
    RETURN_TYPES = ("OBJECT","STRING","STRING")

    def inference(self,model,input, **inputs):
        prompt = f"STORY LOGLINE/SYNOPSIS: {input} \n Based on this story's summary generate the first scene in it's plotline \n FIRST SCENE SUMMARY:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        return (model,response.choices[0].message.content,response.choices[0].message.content)

class EndScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": False, "forceInput":True}),
                "list": ("STRING", {"multiline": False, "forceInput": True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Sequential"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene","list")
    RETURN_TYPES = ("OBJECT","STRING","STRING")

    def inference(self,model,scene,list, **inputs):
        prompt = f"CURRENT SCENE SUMMARY: {scene} \n Based on this scene summary generate the appropriate ending scene summary.\n ENDING SCENE SUMMARY:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        return (model,response.choices[0].message.content,list+"###\n"+response.choices[0].message.content)

class Negative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": True}),
            },
        }

    CATEGORY = "Kinotron/Sequential"
    FUNCTION = "inference"
    RETURN_NAMES = ("model","scene",)
    RETURN_TYPES = ("OBJECT","STRING",)

    def inference(self,model,scene, **inputs):
        prompt = f"SCENE SUMMARY: {scene} \n Given this scene summary rewrite it but with an opposite outcome. \n OPPOSITE SCENE SUMMARY:"
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": ""},
                {"role": "user",
                 "content": prompt}
            ]
        )
        return (model,response.choices[0].message.content,)

#DAGS
class BinaryChoice:
    @classmethod
    def IS_CHANGED(s):
        return True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_A": ("INT",{"default": 1, "min": 0, "max": 1000, "step": 1}),
                "weight_B": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
                "scene_A": ("STRING",{"forceInput": True}),
                "scene_B": ("STRING",{"forceInput": True}),
            },
        }

    CATEGORY = "Kinotron/DAGs"
    FUNCTION = "select"
    RETURN_NAMES = ("Scene",)
    RETURN_TYPES = ("STRING",)

    def select(self,scene_A,scene_B,weight_A,weight_B, **inputs):
        if weight_A+weight_B==0:
            raise ValueError("At least one weight must be positive!")
        p = weight_A/(weight_A+weight_B)
        if random.uniform(0, 1) < p:
            selected = scene_A
        else:
            selected = scene_B
        return (selected,)

class TernaryChoice:
    @classmethod
    def IS_CHANGED(s):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_A": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
                "weight_B": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
                "weight_C": ("INT", {"default": 1, "min": 0, "max": 1000, "step": 1}),
                "scene_A": ("STRING", {"forceInput": True}),
                "scene_B": ("STRING", {"forceInput": True}),
                "scene_C": ("STRING", {"forceInput": True}),
            },
        }

    CATEGORY = "Kinotron/DAGs"
    FUNCTION = "select"
    RETURN_NAMES = ("Scene",)
    RETURN_TYPES = ("STRING",)

    def select(self, scene_A, scene_B, scene_C, weight_A, weight_B, weight_C, **inputs):
        if weight_A + weight_B + weight_C == 0:
            raise ValueError("At least one weight must be positive!")
        p = weight_A / (weight_A + weight_B + weight_C)
        q = (weight_A + weight_B) / (weight_A + weight_B + weight_C)
        x = random.uniform(0, 1)
        if x < p:
            selected = scene_A
        else:
            if x < q:
                selected = scene_B
            else:
                selected = scene_C
        return (selected,)

#GENERATION
class Prose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": False, "forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Generation"
    FUNCTION = "inference"
    RETURN_NAMES = ("prose",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,scene, **inputs):
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert writer, given the user's summary, write prose narrative text."},
                {"role": "user",
                 "content": scene}
            ]
        )
        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)

class Script:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": False, "forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Generation"
    FUNCTION = "inference"
    RETURN_NAMES = ("script",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,scene, **inputs):
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert screenwriter, given the user's scene summary, write the corresponding film script text in Fountain format."},
                {"role": "user",
                 "content": scene}
            ]
        )
        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)

class ScriptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "list": ("LIST", {"multiline": False, "forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Generation"
    FUNCTION = "inference"
    RETURN_NAMES = ("script",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,list, **inputs):
        for scene in list.items:
            response = model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an expert screenwriter, given the user's scene summary, write the corresponding film script text in Fountain format."},
                    {"role": "user",
                     "content": scene}
                ]
            )
            output = output + response.choices[0].message.content
        #print(response.choices[0].message.content)
        return (output,)

class ProseScript:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OBJECT",),
                "scene": ("STRING", {"multiline": False, "forceInput":True}),
            },
        }

    OUTPUT_NODE = True
    CATEGORY = "Kinotron/Generation"
    FUNCTION = "inference"
    RETURN_NAMES = ("script",)
    RETURN_TYPES = ("STRING",)

    def inference(self,model,scene, **inputs):
        response = model.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are an expert algorithm that transforms prose narrative text into the corresponding film script interpretation, given the user's prose text, write the correct film script text in Fountain format."},
                {"role": "user",
                 "content": scene}
            ]
        )
        print(response.choices[0].message.content)
        return (response.choices[0].message.content,)

NODE_CLASS_MAPPINGS = {
#IO
    "Text Input": TextInput,
    "To Console": ToConsole,
    "To Console (List)": ToConsoleList,
    "Combine": Combine,
    #"Save Text": SaveText,
#Lists
    "Empty List": EmptyList,
    "Append": Append,
    "Append (2)": Append2,
    "Append (3)": Append3,
    "Append (5)": Append5,

#Basic inference
    "Load OpenAI": LoadOpenAI,
    "Inference": Inference,
    "Inference (Input)": InferenceTextInput,
#ScriptGeneration
    "Generate Prose": Prose,
    "Generate Script": Script,
    "Genere Script (List)": ScriptList,
    "Prose-to-Script": ProseScript,
#Sequential
    "First Scene": FirstScene,
    "Next Scene": NextScene,
    "Ending": EndScene,
#Tree
    "Expand (2)": Expand2,
    "Expand (3)": Expand3,
    "Interpolate": Interpolate,
#DAG
    "Negative": Negative,
    "Random (2)": BinaryChoice,
    "Random (3)": TernaryChoice,
#Archetypes
    "3-Act": ThreeAct,
    "Freytag": Freytag,
    #"Monomyth": Monomyth,
}