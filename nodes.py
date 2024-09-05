import re
import numpy as np
import torch
from collections import defaultdict
import cv2
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from skimage.color import rgb2lab, deltaE_ciede2000

class prompt_sorting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": ""}),
                "remove_duplicate_tags": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = "advanced/AI_Assistant/text"
    

    def execute(self,string,remove_duplicate_tags):
         return (self.tidy_tags(string,remove_duplicate_tags),)
    
    def tidy_tags(self,text, remove_duplicate_tags):
        if not text or not isinstance(text,(str,list)):
            return ""
        
        # Convert list to string
        if isinstance(text,list):
            text = ",".join(map(str, text))

        
        while True:
            # Eliminate redundant commas and strip whitespace around commas
            new_text = re.sub(r"[,\s\t\n]*,[,\s\t\n]*", ",", text)

            # Fix double spaces
            new_text = re.sub(r"\s\s+", " ", new_text)
            if new_text == text:
                break
            text = new_text

        text = text.strip(" ,\t\r\n")

        # Remove duplicate tags, keeping the duplicate closest to the beginning of the prompt
        if remove_duplicate_tags:
            tags = text.split(',')
            seen = set()
            result = []
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    result.append(tag)
            text = ",".join(result)

        # Clean up BREAKs
        # text = re.sub(r"[,\s\n]*BREAK[,\s\n]*"," BREAK ", text)
        return text


class apply_lighting_effects:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "light_yaw": ("FLOAT", {
                    "default": 60,
                    "min": -180,
                    "max": 180,
                    "step": 0.1,
                    "display": "number"}),
                "light_pitch": ("FLOAT", {
                    "default": -60,
                    "min": -90,
                    "max": 90,
                    "step": 0.1,
                    "display": "number"}),
                "specular_power": ("FLOAT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 0.01,
                    "display": "number"}),
                "normal_diffuse_strength": ("FLOAT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 0.01,
                    "display": "number"}),
                "specular_highlights_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0,
                    "max": 5,
                    "step": 0.01,
                    "display": "number"}),
                "total_gain": ("FLOAT", {
                    "default": 0.6,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "number"}),
                # "select_lighting_option": (
                #     [
                #         'Directly Above',
                #         'Directly Below',
                #         'Directly Left',
                #         'Directly Right',
                #         'Upper Left Diagonal',
                #         'Upper Right Diagonal'
                #     ], {
                #         "default": 'Directly Above'
                # }),
            },

        }

    RETURN_TYPES = ("IMAGE","STRING", )
    RETURN_NAMES = ("image","show_help",)
    FUNCTION = "process"
    CATEGORY = "advanced/AI_Assistant/image"
    DESCRIPTION = """
光源方向參考參數如下:
光源正前方=light_yaw:60 light_pitch:-60
光源左上方=light_yaw:40 light_pitch:-60
光源右上方=light_yaw:60 light_pitch:-40
光源左=light_yaw:0 light_pitch:0
光源右=light_yaw:90 light_pitch:0
光源下方=light_yaw:45 light_pitch:0
"""


    def process(self, image, light_yaw, light_pitch, specular_power, normal_diffuse_strength,
                 specular_highlights_strength, total_gain):
        show_help ="光源方向參考參數如下:\n光源正前方=light_yaw:60 light_pitch:-60\n光源左上方=light_yaw:40 light_pitch:-60\n光源右上方=light_yaw:60 light_pitch:-40\n光源左=light_yaw:0 light_pitch:0\n光源右=light_yaw:90 light_pitch:0\n光源下方=light_yaw:45 light_pitch:0\n"
        
        return (self.lighting_effects(image, light_yaw, light_pitch, specular_power, normal_diffuse_strength,
                 specular_highlights_strength, total_gain),show_help,)
        
    def lighting_effects(self, input_tensor, light_yaw, light_pitch, specular_power, normal_diffuse_strength,
                                specular_highlights_strength, total_gain):

            # 元の次元数を記録する
            original_dim = input_tensor.dim()
            #print(original_dim,input_tensor.shape,input_tensor.shape[1])

            # 入力テンソルが最低でも4次元であることを保証する（バッチ次元がなければ追加する）
            if original_dim == 3:
                input_tensor = input_tensor.unsqueeze(0)  # バッチ次元を追加
                
            # 轉換形狀為 (Batch, Channels, Height, Width)
            input_tensor = input_tensor.permute(0, 3, 1, 2)
            # 入力テンソルに3つ以上のチャンネルがある場合、追加のチャンネルをアルファと仮定し削除する
            if input_tensor.shape[1] > 3:
                input_tensor = input_tensor[:, :3, :, :]  # 最初の3チャンネルのみを保持

            # 入力テンソルを正規化して法線ベクトルを取得する
            normal_tensor = torch.nn.functional.normalize(input_tensor, dim=1)  # チャンネル次元に沿って正規化

            # 光の方向ベクトルを計算する
            light_direction = self.euler_to_vector(light_yaw, light_pitch, 0)
            light_direction = light_direction.to(input_tensor.device)  # 同じデバイスを保証する

            # 光の方向とカメラの方向を入力テンソルの次元に合わせて拡張する
            batch_size, _, height, width = input_tensor.shape
            light_direction = light_direction.view(1, 3, 1, 1).expand(batch_size, -1, height, width)
            camera_direction = torch.tensor([0, 0, 1], dtype=torch.float32, device=input_tensor.device)
            camera_direction = camera_direction.view(1, 3, 1, 1).expand(batch_size, -1, height, width)

            # 拡散成分を計算する
            diffuse = torch.sum(normal_tensor * light_direction, dim=1, keepdim=True)
            diffuse = torch.clamp(diffuse, 0, 1)

            # 鏡面成分を計算する
            half_vector = torch.nn.functional.normalize(light_direction + camera_direction, dim=1)
            specular = torch.sum(normal_tensor * half_vector, dim=1, keepdim=True)
            specular = torch.pow(torch.clamp(specular, 0, 1), specular_power)

            # 拡散成分と鏡面成分を組み合わせて、強度とゲインを適用する
            output_tensor = (diffuse * normal_diffuse_strength + specular * specular_highlights_strength) * total_gain
            output_tensor = output_tensor.squeeze(1)  # keepdim=Trueで追加されたチャンネル次元を削除

            # 初めに追加されたバッチ次元があれば削除する
            if original_dim == 3:
                output_tensor = output_tensor.squeeze(0)

            return output_tensor

    def euler_to_vector(self, yaw, pitch, roll):
        # オイラー角から方向ベクトルを計算
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        direction = np.array([
            sin_yaw * cos_pitch,
            -sin_pitch,
            cos_yaw * cos_pitch
        ])
        return torch.from_numpy(direction).float()
    
class clean_prompt_tags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "Clean_Prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("string","show_help")
    FUNCTION = "execute"
    CATEGORY = "advanced/AI_Assistant/text"
    DESCRIPTION = """
粗略的移除工具:

EX:
base_prompt = masterpiece, best quality, monochrome, greyscale,blue_hair, blue_sky, blue_dress, pink_eyes, pink_bow, pink_panties

removeprompt = pink, red, orange, brown, yellow, green, blue, purple, blonde, colored skin, white hair,

輸出則為:masterpiece, best quality, monochrome, greyscale
"""

    def execute(self,string,Clean_Prompt):
         show_help ="粗略的移除工具:\n\nEX:\n\nbase_prompt = masterpiece, best quality, monochrome, greyscale,blue_hair, blue_sky, blue_dress, pink_eyes, pink_bow, pink_panties\n\nremoveprompt = pink, red, orange, brown, yellow, green, blue, purple, blonde, colored skin, white hair,\n\n輸出則為:\nmasterpiece, best quality, monochrome, greyscale"
         return (self.remove_prompt(string,Clean_Prompt),show_help)
    
    def remove_prompt(self,base_prompt, remove_prompt,):
        
        # 移除末尾多餘的逗號並使用正規表達式來分割標籤
        remove_prompt = re.sub(r',\s*$', '', remove_prompt).replace('\n', '')

        # 使用正規表達式來分割標籤，忽略逗號後的多餘空格
        prompt_list = re.split(r',\s*', base_prompt)
        color_list = re.split(r',\s*', remove_prompt)

        # カラータグを除去します。
        cleaned_tags = [tag for tag in prompt_list if all(color.lower() not in tag.lower() for color in color_list)]

        return ", ".join(cleaned_tags)

class prompt_blacklist:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "Blacklist_Prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("string","show_help")
    FUNCTION = "execute"
    CATEGORY = "advanced/AI_Assistant/text"
    DESCRIPTION = """
黑名單功能，分別輸入需移除的提示以逗號分隔即可

EX:
base_prompt= masterpiece, best quality,lineart,sketch,transparent background
Blacklist Prompt=lineart,sketch,
輸出則為:masterpiece, best quality, transparent background    
"""
    def execute(self,Blacklist_Prompt,string):
         show_help ="黑名單功能，分別輸入需移除的提示以逗號分隔即可\n\nEX:\nbase_prompt= masterpiece, best quality,lineart,sketch,transparent background\nBlacklist Prompt=lineart,sketch,\n輸出則為:\n masterpiece, best quality, transparent background"
         return (self.execute_prompt(Blacklist_Prompt,string),show_help)
    
    def execute_prompt(self,execute_tags, base_prompt):
        prompt_list = re.split(r',\s*', base_prompt)
        # execute_tagsを除去
        filtered_tags = [tag for tag in prompt_list if tag not in execute_tags]
        # 最終的なプロンプトを生成
        return ", ".join(filtered_tags)

class resize_image_sdxl_ratio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT","INT","STRING",)
    RETURN_NAMES = ("Target Width","Target Height","show_help")
    FUNCTION = "execute"
    CATEGORY = "advanced/AI_Assistant/image"
    DESCRIPTION = """
自動輸出對應的SDXL目標尺寸與ImageScale(圖像縮放)搭配使用並開啟剪裁功能
    """
    def execute(self,image):
         width,height=self.resize_image_aspect_ratio(image)
         show_help ="自動輸出對應的SDXL目標尺寸與ImageScale(圖像縮放)搭配使用並開啟剪裁功能"
         return (width, height, show_help)
    
    def ResolutionFromImage(self, image):
        _, H, W, _ = image.shape
        return (W, H)
    
    def resize_image_aspect_ratio(self,image):
        # 元の画像サイズを取得
        original_width, original_height = self.ResolutionFromImage(image)

        # アスペクト比を計算
        aspect_ratio = original_width / original_height

        # 標準のアスペクト比サイズを定義
        sizes = {
            1: (1024, 1024),  # 正方形
            4/3: (1152, 896),  # 横長画像
            3/2: (1216, 832),
            16/9: (1344, 768),
            21/9: (1568, 672),
            3/1: (1728, 576),
            1/4: (512, 2048),  # 縦長画像
            1/3: (576, 1728),
            9/16: (768, 1344),
            2/3: (832, 1216),
            3/4: (896, 1152)
        }

        # 最も近いアスペクト比を見つける
        closest_aspect_ratio = min(sizes.keys(), key=lambda x: abs(x - aspect_ratio))
        target_width, target_height = sizes[closest_aspect_ratio]

        return (target_width, target_height)
    
class noline_process:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },

        }

    RETURN_TYPES = ("IMAGE","STRING", )
    RETURN_NAMES = ("image","show_help",)
    FUNCTION = "process"
    CATEGORY = "advanced/AI_Assistant/image"
    DESCRIPTION = """
將輸入圖像移除線條並將顏色取近似值用於I2I
    """

    def process(self, image,):
        show_help ="將輸入圖像移除線條並將顏色取近似值用於I2I\n"
        
        return (self.noline_process(image),show_help,)
        
    def noline_process(self, input_image):

        #ToPILImage
        input_image = input_image.permute(0, 3, 1, 2)
        input_image = input_image.squeeze(0)
        input_image = transforms.ToPILImage()(input_image)

        def get_major_colors(image, threshold_percentage):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            color_count = defaultdict(int)
            for pixel in image.getdata():
                color_count[pixel] += 1
            total_pixels = image.width * image.height
            return [(color, count) for color, count in color_count.items() if (count / total_pixels) >= threshold_percentage]

        def consolidate_colors(major_colors, threshold):
            colors_lab = [rgb2lab(np.array([[color]], dtype=np.float32)/255.0).reshape(3) for color, _ in major_colors]
            i = 0
            while i < len(colors_lab):
                j = i + 1
                while j < len(colors_lab):
                    if deltaE_ciede2000(colors_lab[i], colors_lab[j]) < threshold:
                        if major_colors[i][1] >= major_colors[j][1]:
                            major_colors[i] = (major_colors[i][0], major_colors[i][1] + major_colors[j][1])
                            major_colors.pop(j)
                            colors_lab.pop(j)
                        else:
                            major_colors[j] = (major_colors[j][0], major_colors[j][1] + major_colors[i][1])
                            major_colors.pop(i)
                            colors_lab.pop(i)
                        continue
                    j += 1
                i += 1
            return major_colors

        def generate_distant_colors(consolidated_colors, distance_threshold):
            consolidated_lab = [rgb2lab(np.array([color], dtype=np.float32) / 255.0).reshape(3) for color, _ in consolidated_colors]
            max_attempts = 10000
            for _ in range(max_attempts):
                random_rgb = np.random.randint(0, 256, size=3)
                random_lab = rgb2lab(np.array([random_rgb], dtype=np.float32) / 255.0).reshape(3)
                if all(deltaE_ciede2000(base_color_lab, random_lab) > distance_threshold for base_color_lab in consolidated_lab):
                    return tuple(random_rgb)
            return (128, 128, 128)

        def line_color(image, mask, new_color):
            data = np.array(image)
            data[mask, :3] = new_color
            return Image.fromarray(data)

        def replace_color(image, color_1, blur_radius=2):
            data = np.array(image)
            original_shape = data.shape
            channels = original_shape[2] if len(original_shape) > 2 else 1
            data = data.reshape(-1, channels)
            color_1 = np.array(color_1)
            matches = np.all(data[:, :3] == color_1, axis=1)
            mask = np.zeros(data.shape[0], dtype=bool)

            while np.any(matches):
                new_matches = np.zeros_like(matches)
                for i in range(len(data)):
                    if matches[i]:
                        x, y = divmod(i, original_shape[1])
                        neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                        valid_neighbors = [data[nx * original_shape[1] + ny, :3] for nx, ny in neighbors if 0 <= nx < original_shape[0] and 0 <= ny < original_shape[1] and not matches[nx * original_shape[1] + ny]]
                        if valid_neighbors:
                            new_color = np.mean(valid_neighbors, axis=0).astype(np.uint8)
                            data[i, :3] = new_color
                            mask[i] = True
                        else:
                            new_matches[i] = True
                matches = new_matches
                if not np.any(matches):
                    break

            data = data.reshape(original_shape)
            mask = mask.reshape(original_shape[:2])
            result_image = Image.fromarray(data, 'RGBA')
            blurred_image = result_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_data = np.array(blurred_image)
            np.copyto(data, blurred_data, where=mask[..., None])
            return Image.fromarray(data, 'RGBA')

        def DoG_filter(image, kernel_size=0, sigma=1.0, k_sigma=2.0, gamma=1.5):
            g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
            return g1 - gamma * g2

        def XDoG_filter(image, kernel_size=0, sigma=1.4, k_sigma=1.6, epsilon=0, phi=10, gamma=0.98):
            epsilon /= 255
            dog = DoG_filter(image, kernel_size, sigma, k_sigma, gamma)
            dog /= dog.max()
            e = 1 + np.tanh(phi * (dog - epsilon))
            e[e >= 1] = 1
            return (e * 255).astype('uint8')

        def binarize_image(image):
            _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binarized


        def process_XDoG(image):
            kernel_size = 0
            sigma = 1.4
            k_sigma = 1.6
            epsilon = 0
            phi = 10
            gamma = 0.98

            # 將 PIL Image 對象轉換為灰度模式的 numpy 數組
            image_array = np.array(image.convert("L"))
            xdog_image = XDoG_filter(image_array, kernel_size, sigma, k_sigma, epsilon, phi, gamma)
            binarized_image = binarize_image(xdog_image)
            final_image = Image.fromarray(binarized_image)
            return final_image
        
        rgb_image = input_image.convert("RGBA")
        lineart = process_XDoG(input_image).convert('L')
        lineart = lineart.point(lambda x: 0 if x < 200 else 255)
        lineart = ImageOps.invert(lineart)
        kernel = np.ones((3, 3), np.uint8)
        lineart = cv2.dilate(np.array(lineart), kernel, iterations=1)
        lineart = Image.fromarray(lineart)
        mask = np.array(lineart) == 255
        major_colors = get_major_colors(rgb_image, threshold_percentage=0.05)
        major_colors = consolidate_colors(major_colors, 10)
        new_color_1 = generate_distant_colors(major_colors, 100)
        filled_image = line_color(rgb_image, mask, new_color_1)
        replace_color_image = replace_color(filled_image, new_color_1, 2).convert('RGB')
        #ToTensor
        replace_color_image=transforms.ToTensor()(replace_color_image)
        replace_color_image = replace_color_image.unsqueeze(0)
        replace_color_image = replace_color_image.permute(0, 2, 3, 1)
        #print(replace_color_image.shape)
        return replace_color_image
