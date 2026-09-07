#predict_ellipse7-2の類似　カメラ追加

import random
import numpy as np
from ultralytics import YOLO
import csv
import random
import cv2
from scipy import optimize
from PIL import Image
import os
import gdown

def preprocess_640(img):
    """PIL画像を受け取り、中心正方形にトリミング→640×640にリサイズして返す。
    （resize.py と同じロジックを、ファイル入出力なしで関数化したもの）"""
    # RGBに統一（PNGのalpha付きやグレースケール画像対策）
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    aspect_ratio = width / height

    # 短辺に合わせた正方形サイズ
    if aspect_ratio > 1:      # 横長 → 高さに合わせる
        new_width = height
        new_height = height
    else:                     # 縦長 → 幅に合わせる
        new_width = width
        new_height = width

    # 中心を基準にトリミング
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img_cropped = img.crop((left, top, right, bottom))

    # 640×640にリサイズ
    return img_cropped.resize((640, 640))


def is_point_on_side(p, p1, p2, left=True):
    """2つの座標を結ぶ直線と他の点との位置関係を計算"""
    return ((p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) > 0) == left

def get_middle_point(points):
    """複数の座標点の中心点を計算"""
    return np.mean(points, axis=0).astype(int)

def fit_ellipse_from_points(points):
    """点群から楕円を近似し、中心・長軸長・短軸長・長軸方向ベクトルを返す。
    points : shape (N, 2) の配列（N >= 5 必要）
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 5:
        raise ValueError("楕円フィッティングには最低5点必要です")

    # cv2.fitEllipse の返り値: (center, (d1, d2), angle)
    #   d1, d2 : 2軸の全長（直径）。大小はソートされていない
    #   angle  : d1 軸の向き [度]
    ellipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    (cx, cy), (d1, d2), angle = ellipse

    # 大きい方を長軸、小さい方を短軸として確定
    if d1 >= d2:
        major_length, minor_length = d1, d2
        major_angle_deg = angle
    else:
        major_length, minor_length = d2, d1
        major_angle_deg = angle + 90.0  # d2 軸は d1 軸に直交

    # 長軸方向の単位ベクトル（画像座標系: x右・y下）
    theta = np.radians(major_angle_deg)
    major_vector = np.array([np.cos(theta), np.sin(theta)])

    return {
        "center": (cx, cy),
        "major_length": major_length,
        "minor_length": minor_length,
        "major_angle_deg": major_angle_deg,
        "major_vector": major_vector,
        "cv2_ellipse": ellipse,  # 描画用に生タプルも保持
    }

def calculate_anteversion_angle(cup_data, boundary_points):
    """前方開角を計算（輪郭5点を楕円近似し、Lewinnek法 arcsin(短軸/長軸)）"""
    # 4隅の極点もバウンディングボックス対角線も使わず、
    # 輪郭から抽出した境界点（5点）だけで楕円を近似する
    points = []
    for point in boundary_points:
        if point['point_x'] is not None and point['point_y'] is not None:
            points.append([point['point_x'], point['point_y']])
    points = np.array(points)

    # 楕円を近似
    fit = fit_ellipse_from_points(points)

    # 検証用: 近似結果を出力（後で消してOK）
    print(f"  [anteversion fit] 長軸={fit['major_length']:.1f} "
          f"短軸={fit['minor_length']:.1f} "
          f"短軸/長軸={fit['minor_length']/fit['major_length']:.3f}")

    # Lewinnek法: 前方開角 = arcsin(短軸 / 長軸)
    anteversion_angle = np.degrees(
        np.arcsin(fit['minor_length'] / fit['major_length'])
    )
    return anteversion_angle

def calculate_abduction_angle(boundary_points, obturator_line):
    """外方開角を計算（フィット楕円の長軸ベクトルと閉鎖孔基準線のなす角）"""
    # 輪郭5点で楕円を近似し、その長軸方向をカップ基準線とする
    points = []
    for point in boundary_points:
        if point['point_x'] is not None and point['point_y'] is not None:
            points.append([point['point_x'], point['point_y']])
    points = np.array(points)

    fit = fit_ellipse_from_points(points)
    cup_vector = fit['major_vector']

    obturator_vector = np.array([
        obturator_line['right_x'] - obturator_line['left_x'],
        obturator_line['right_y'] - obturator_line['left_y']
    ])

    # 検証用（後で消してOK）
    print(f"  [abduction] 長軸方向="
          f"{np.degrees(np.arctan2(cup_vector[1], cup_vector[0])):.1f}° "
          f"閉鎖孔線方向="
          f"{np.degrees(np.arctan2(obturator_vector[1], obturator_vector[0])):.1f}°")

    dot_product = np.dot(cup_vector, obturator_vector)
    norms = np.linalg.norm(cup_vector) * np.linalg.norm(obturator_vector)
    angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))

    if angle > 90:
        angle = 180 - angle

    return angle

def calculate_leg_length(obturator_point, trochanter_point, obturator_line_length):
    """
    脚長を垂直距離として計算し、閉鎖孔の基準線の距離で正規化
    """
    # 垂直距離を計算（y座標の差の絶対値）
    vertical_distance = abs(obturator_point['top_point_y'] - trochanter_point['top_point_y'])
    
    # 脚長を正規化（基準線の長さに対する比率を100倍）
    normalized_length = (100 * vertical_distance) / obturator_line_length if obturator_line_length > 0 else None
    
    # 垂直線を描画するための座標を計算
    vertical_line = {
        'start_x': trochanter_point['top_point_x'],
        'start_y': trochanter_point['top_point_y'],
        'end_x': trochanter_point['top_point_x'],
        'end_y': obturator_point['top_point_y']
    }
    
    return normalized_length, vertical_line

def calculate_obturator_line_length(left_obturator, right_obturator):
    """閉鎖孔の基準線の長さを計算"""
    dx = right_obturator['top_point_x'] - left_obturator['top_point_x']
    dy = right_obturator['top_point_y'] - left_obturator['top_point_y']
    return np.sqrt(dx**2 + dy**2)

def process_measurements(saved_coordinates):
    """全ての測定を実行"""
    measurements = {}
    
    cup_ellipses = {}
    cup_boundaries = {}
    obturator_points = {}
    trochanter_points = {}
    
    for coord in saved_coordinates:
        class_name = coord['class_name']
        
        if class_name in ['right_cup_ellipse', 'left_cup_ellipse']:
            cup_ellipses[class_name] = coord
        elif '_boundary_' in class_name:
            base_name = class_name.split('_boundary_')[0]
            if base_name not in cup_boundaries:
                cup_boundaries[base_name] = []
            cup_boundaries[base_name].append(coord)
        elif 'obturator_foramen' in class_name:
            side = class_name.split('_')[0]
            obturator_points[side] = coord
        elif 'lesser_trochanter' in class_name:
            side = class_name.split('_')[0]
            trochanter_points[side] = coord
    
    # 前方開角の計算
    print("\n=== 前方開角（Anteversion Angle）===")
    for side in ['right', 'left']:
        cup_name = f'{side}_cup_ellipse'
        if cup_name in cup_ellipses and cup_name in cup_boundaries:
            angle = calculate_anteversion_angle(
                cup_ellipses[cup_name],
                cup_boundaries[cup_name]
            )
            measurements[f'{side}_anteversion_angle'] = angle
            print(f"{side.capitalize()}側: {angle:.1f}°")
        else:
            print(f"{side.capitalize()}側: 測定不可")
    
    # 外方開角の計算
    print("\n=== 外方開角（Abduction Angle）===")
    if 'left' in obturator_points and 'right' in obturator_points:
        obturator_line = {
            'left_x': obturator_points['left']['top_point_x'],
            'left_y': obturator_points['left']['top_point_y'],
            'right_x': obturator_points['right']['top_point_x'],
            'right_y': obturator_points['right']['top_point_y']
        }
        
        for side in ['right', 'left']:
            cup_name = f'{side}_cup_ellipse'
            if cup_name in cup_ellipses and cup_name in cup_boundaries:
                angle = calculate_abduction_angle(
                    cup_boundaries[cup_name],
                    obturator_line
                )
                measurements[f'{side}_abduction_angle'] = angle
                print(f"{side.capitalize()}側: {angle:.1f}°")
            else:
                print(f"{side.capitalize()}側: 測定不可")
    else:
        print("閉鎖孔の検出が不完全なため測定不可")

    # 閉鎖孔の基準線の長さを計算
    if 'left' in obturator_points and 'right' in obturator_points:
        obturator_line_length = calculate_obturator_line_length(
            obturator_points['left'],
            obturator_points['right']
        )
        measurements['obturator_line_length'] = obturator_line_length
    else:
        obturator_line_length = None
    
    # 脚長の計算
    print("\n=== 脚長（Normalized Leg Length）===")
    leg_lengths = {}
    vertical_lines = {}
    
    for side in ['right', 'left']:
        if side in obturator_points and side in trochanter_points and obturator_line_length:
            length, vertical_line = calculate_leg_length(
                obturator_points[side],
                trochanter_points[side],
                obturator_line_length
            )
            if length is not None:
                measurements[f'{side}_leg_length'] = length
                measurements[f'{side}_vertical_line'] = vertical_line
                leg_lengths[side] = length
                vertical_lines[side] = vertical_line
                print(f"{side.capitalize()}側: {length:.1f}mm")
        else:
            print(f"{side.capitalize()}側: 測定不可")
    
    if len(leg_lengths) == 2:
        difference = abs(leg_lengths['right'] - leg_lengths['left'])
        longer_side = 'Right' if leg_lengths['right'] > leg_lengths['left'] else 'Left'
        print(f"\n脚長差: {difference:.1f}mm ({longer_side}側が長い)")
    
    return measurements

def visualize_measurements(image, saved_coordinates, measurements):
    """測定結果の可視化"""
    image = image.copy()
    
    cup_ellipses = {}
    cup_boundaries = {}
    obturator_points = {}
    trochanter_points = {}
    
    for coord in saved_coordinates:
        class_name = coord['class_name']
        
        if class_name in ['right_cup_ellipse', 'left_cup_ellipse']:
            cup_ellipses[class_name] = coord
        elif '_boundary_' in class_name:
            base_name = class_name.split('_boundary_')[0]
            if base_name not in cup_boundaries:
                cup_boundaries[base_name] = []
            cup_boundaries[base_name].append(coord)
        elif 'obturator_foramen' in class_name:
            side = class_name.split('_')[0]
            obturator_points[side] = {
                'top_point_x': coord['top_point_x'],
                'top_point_y': coord['top_point_y']
            }
        elif 'lesser_trochanter' in class_name:
            side = class_name.split('_')[0]
            trochanter_points[side] = {
                'top_point_x': coord['top_point_x'],
                'top_point_y': coord['top_point_y']
            }

    # カップの可視化
    for side in ['right', 'left']:
        cup_name = f'{side}_cup_ellipse'
        if cup_name in cup_ellipses and cup_name in cup_boundaries:
            # 輪郭5点だけを集める
            points = []
            for point in cup_boundaries[cup_name]:
                if point['point_x'] is not None and point['point_y'] is not None:
                    points.append([point['point_x'], point['point_y']])
            points = np.array(points)

            # 楕円を近似
            fit = fit_ellipse_from_points(points)
            center = (int(fit['center'][0]), int(fit['center'][1]))
            major_vector = fit['major_vector']
            half_major = fit['major_length'] / 2.0

            # 長軸（カップ基準線）の始点・終点 = 中心 ± 半長軸 × 長軸方向
            start = (int(center[0] - half_major * major_vector[0]),
                     int(center[1] - half_major * major_vector[1]))
            end = (int(center[0] + half_major * major_vector[0]),
                   int(center[1] + half_major * major_vector[1]))

            # 使用した5点を描画（赤色）
            for point in points:
                cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            # フィッティングした楕円を描画（青色）
            cv2.ellipse(image, fit['cv2_ellipse'], (255, 0, 0), 2)

            # 長軸 = カップ基準線を描画（黄色）
            cv2.line(image, start, end, (0, 255, 255), 2)
    
    # 閉鎖孔と外方開角の可視化
    if 'left' in obturator_points and 'right' in obturator_points:
        # 閉鎖孔を結ぶ線を描画（緑色）
        cv2.line(image, 
                 (obturator_points['left']['top_point_x'], obturator_points['left']['top_point_y']),
                 (obturator_points['right']['top_point_x'], obturator_points['right']['top_point_y']),
                 (0, 255, 0), 2)
    
    # 脚長の可視化
    for side in ['right', 'left']:
        if (f'{side}_leg_length' in measurements and 
            f'{side}_vertical_line' in measurements):
            # 最上点を強調表示（黄色の点）
            cv2.circle(image, 
                      (obturator_points[side]['top_point_x'], 
                       obturator_points[side]['top_point_y']), 
                      5, (0, 255, 255), -1)  # 閉鎖孔の最上点
            cv2.circle(image, 
                      (trochanter_points[side]['top_point_x'], 
                       trochanter_points[side]['top_point_y']), 
                      5, (0, 255, 255), -1)  # 小転子の最上点
            
            # 垂直線を描画（シアン色）
            vertical_line = measurements[f'{side}_vertical_line']
            cv2.line(image,
                     (vertical_line['start_x'], vertical_line['start_y']),
                     (vertical_line['end_x'], vertical_line['end_y']),
                     (255, 255, 0), 2)
            
            # 水平の補助線を描画（点線、白色）
            cv2.line(image,
                     (obturator_points[side]['top_point_x'], 
                      obturator_points[side]['top_point_y']),
                     (vertical_line['end_x'], vertical_line['end_y']),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # 測定値を画像に描画
    y_pos = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    for side in ['right', 'left']:
        # 前方開角
        if f'{side}_anteversion_angle' in measurements:
            cv2.putText(image, f"{side.capitalize()} Anteversion: {measurements[f'{side}_anteversion_angle']:.1f}°",
                       (10, y_pos), font, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # 外方開角
        if f'{side}_abduction_angle' in measurements:
            cv2.putText(image, f"{side.capitalize()} Abduction: {measurements[f'{side}_abduction_angle']:.1f}°",
                       (10, y_pos), font, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # 正規化された脚長
        if f'{side}_leg_length' in measurements:
            cv2.putText(image, f"{side.capitalize()} Leg Length: {measurements[f'{side}_leg_length']:.1f}mm",
                       (10, y_pos), font, 0.7, (255, 255, 255), 2)
            y_pos += 30

    # 可視化した画像を返す（保存はしない）
    return image

def run_measurement(model, pil_image):
    # 前処理：640×640に整え、OpenCV用のBGR配列にする
    pil_640 = preprocess_640(pil_image)
    image_bgr = cv2.cvtColor(np.array(pil_640), cv2.COLOR_RGB2BGR)

        # 画像に対して推論を実行
    print("画像の推論を実行中...")
    results = model.predict(source=image_bgr, show=False, save=False, show_labels=False, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=1, show_boxes=True)

    # 検出結果の振り分け
    print("検出結果をフィルタリング中...")
    filtered_results = {}  # cup以外のクラスで最も信頼度の高い検出結果を保存
    cups = []  # cupとcup_ellipseの全検出結果を保存

    for box, mask, cls in zip(results[0].boxes.data, results[0].masks.data, results[0].boxes.cls):
        class_id = int(cls)
        conf = box[4].item()  # 信頼度スコア
        class_name = results[0].names[class_id]
        
        if class_name in ['cup', 'cup_ellipse']:
            cups.append({
                'mask': mask,
                'class_id': class_id,
                'conf': conf,
                'class_name': class_name
            })
            print(f"{class_name}を検出: 信頼度 {conf:.3f}")
        else:
            if class_name not in filtered_results or conf > filtered_results[class_name]['conf']:
                filtered_results[class_name] = {
                    'mask': mask,
                    'class_id': class_id,
                    'conf': conf
                }
                print(f"クラス {class_name} の検出を更新: 信頼度 {conf:.3f}")

    saved_coordinates = []

    # cup以外のクラスの処理
    print("\n通常オブジェクトの座標抽出を開始...")
    for class_name, data in filtered_results.items():
        print(f"\n{class_name} の処理中...")
        mask = data['mask']
        class_id = data['class_id']
        
        # マスクをnumpy配列に変換
        mask_np = mask.cpu().numpy()
        
        # マスク内の物体が存在するピクセルの座標を取得
        mask_indices = np.argwhere(mask_np > 0)
        
        if mask_indices.size > 0:
            # 最上、最下、最右、最左の座標を取得
            y_min_idx = np.argmin(mask_indices[:, 0])
            y_max_idx = np.argmax(mask_indices[:, 0])
            x_min_idx = np.argmin(mask_indices[:, 1])
            x_max_idx = np.argmax(mask_indices[:, 1])
            
            # インデックスを座標に変換
            top_point = mask_indices[y_min_idx][::-1]
            bottom_point = mask_indices[y_max_idx][::-1]
            left_point = mask_indices[x_min_idx][::-1]
            right_point = mask_indices[x_max_idx][::-1]

            # 特殊な処理（左閉鎖孔、右閉鎖孔、左小転子、右小転子）
            if class_name == 'left_obturator_foramen':
                print("左閉鎖孔の最上点を調整中...")
                y_min = np.min(mask_indices[:, 0])
                top_points = mask_indices[mask_indices[:, 0] == y_min]
                top_point = top_points[np.argmax(top_points[:, 1])][::-1]
            
            elif class_name == 'right_obturator_foramen':
                print("右閉鎖孔の最上点を調整中...")
                y_min = np.min(mask_indices[:, 0])
                top_points = mask_indices[mask_indices[:, 0] == y_min]
                top_point = top_points[np.argmin(top_points[:, 1])][::-1]
            
            elif class_name == 'left_lesser_trochanter' or class_name == 'right_lesser_trochanter':
                print(f"{class_name}の最上点を調整中...")
                y_min = np.min(mask_indices[:, 0])
                top_points = mask_indices[mask_indices[:, 0] == y_min]
                if len(top_points) > 1:
                    top_point = get_middle_point(top_points)[::-1]
                else:
                    top_point = top_points[0][::-1]
                    
                # 他の特徴点も保持するために取得
                y_max = np.max(mask_indices[:, 0])
                bottom_points = mask_indices[mask_indices[:, 0] == y_max]
                bottom_point = get_middle_point(bottom_points)[::-1]
                
                x_min = np.min(mask_indices[:, 1])
                left_points = mask_indices[mask_indices[:, 1] == x_min]
                left_point = get_middle_point(left_points)[::-1]
                
                x_max = np.max(mask_indices[:, 1])
                right_points = mask_indices[mask_indices[:, 1] == x_max]
                right_point = get_middle_point(right_points)[::-1]

            # 座標を保存
            print(f"{class_name} の特徴点を保存中...")
            saved_coordinates.append({
                "class_name": class_name,
                "top_point_x": int(top_point[0]),
                "top_point_y": int(top_point[1]),
                "bottom_point_x": int(bottom_point[0]),
                "bottom_point_y": int(bottom_point[1]),
                "left_point_x": int(left_point[0]),
                "left_point_y": int(left_point[1]),
                "right_point_x": int(right_point[0]),
                "right_point_y": int(right_point[1]),
                "point_x": None,
                "point_y": None,
                "top_left_x": None,
                "top_left_y": None,
                "top_right_x": None,
                "top_right_y": None,
                "bottom_left_x": None,
                "bottom_left_y": None,
                "bottom_right_x": None,
                "bottom_right_y": None
            })
        else:
            print(f"物体: {class_name} - マスクが見つかりません")

    # カップとカップ楕円の処理
    print("\nカップとカップ楕円の座標抽出を開始...")
    for i, cup_data in enumerate(cups):
        print(f"\nカップ関連オブジェクト {i+1} を処理中...")
        mask = cup_data['mask']
        original_class_name = cup_data['class_name']
        mask_np = mask.cpu().numpy()
        mask_indices = np.argwhere(mask_np > 0)
        
        if mask_indices.size > 0:
            # 最上、最下、最右、最左の座標を取得
            y_min_idx = np.argmin(mask_indices[:, 0])
            y_max_idx = np.argmax(mask_indices[:, 0])
            x_min_idx = np.argmin(mask_indices[:, 1])
            x_max_idx = np.argmax(mask_indices[:, 1])
            
            # インデックスを座標に変換
            top_point = mask_indices[y_min_idx][::-1]
            bottom_point = mask_indices[y_max_idx][::-1]
            left_point = mask_indices[x_min_idx][::-1]
            right_point = mask_indices[x_max_idx][::-1]

            if original_class_name == 'cup_ellipse':
                # バウンディングボックスの座標を取得
                x_min, y_min = np.min(mask_indices[:, 1]), np.min(mask_indices[:, 0])
                x_max, y_max = np.max(mask_indices[:, 1]), np.max(mask_indices[:, 0])
                
                # 左右の判定
                if bottom_point[0] < top_point[0]:
                    class_name = "left_cup_ellipse"
                else:
                    class_name = "right_cup_ellipse"
                
                print(f"{class_name}のバウンディングボックスと特徴点を保存中...")
                saved_coordinates.append({
                    "class_name": class_name,
                    "top_point_x": int(top_point[0]),
                    "top_point_y": int(top_point[1]),
                    "bottom_point_x": int(bottom_point[0]),
                    "bottom_point_y": int(bottom_point[1]),
                    "left_point_x": int(left_point[0]),
                    "left_point_y": int(left_point[1]),
                    "right_point_x": int(right_point[0]),
                    "right_point_y": int(right_point[1]),
                    "point_x": None,
                    "point_y": None,
                    "top_left_x": int(x_min),
                    "top_left_y": int(y_min),
                    "top_right_x": int(x_max),
                    "top_right_y": int(y_min),
                    "bottom_left_x": int(x_min),
                    "bottom_left_y": int(y_max),
                    "bottom_right_x": int(x_max),
                    "bottom_right_y": int(y_max)
                })
                
                # 境界領域の点を抽出（輪郭から「弧長」基準で等間隔N点）
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(contours) > 0:
                    contour = max(contours, key=len).reshape(-1, 2).astype(np.float64)

                    start_idx = random.randint(0, len(contour) - 1)
                    contour = np.roll(contour, -start_idx, axis=0)

                    # 輪郭に沿った累積距離（周長）を計算
                    diffs = np.diff(np.vstack([contour, contour[0]]), axis=0)
                    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
                    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
                    total = cum[-1]

                    # 全周を等分した位置に最も近い輪郭点を選ぶ
                    num_boundary_points = 11  # 楕円近似の点の数
                    targets = np.linspace(0, total, num_boundary_points, endpoint=False)
                    idxs = np.clip(np.searchsorted(cum, targets), 0, len(contour) - 1)
                    selected_points = contour[idxs].astype(int)

                    # 検証用（後で消してOK）
                    print(f"{class_name} の境界{num_boundary_points}点: "
                          f"{[tuple(int(v) for v in p) for p in selected_points]}")

                    for j, point in enumerate(selected_points):
                        saved_coordinates.append({
                            "class_name": f"{class_name}_boundary_{j+1}",
                            "top_point_x": None, "top_point_y": None,
                            "bottom_point_x": None, "bottom_point_y": None,
                            "left_point_x": None, "left_point_y": None,
                            "right_point_x": None, "right_point_y": None,
                            "point_x": int(point[0]),
                            "point_y": int(point[1]),
                            "top_left_x": None, "top_left_y": None,
                            "top_right_x": None, "top_right_y": None,
                            "bottom_left_x": None, "bottom_left_y": None,
                            "bottom_right_x": None, "bottom_right_y": None
                        })

            else:  # 通常のカップ処理
                # カップの左右判定と特別処理
                if bottom_point[0] < top_point[0]:
                    cup_class_name = "left_cup"
                    p1, p2, select_left_side = bottom_point, right_point, True
                else:
                    cup_class_name = "right_cup"
                    p1, p2, select_left_side = bottom_point, left_point, False

                # カップの境界点を抽出
                print(f"{cup_class_name} の境界点を抽出中...")
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    contour = contours[0]
                    num_points = min(len(contour), 50)
                    selected_points = random.sample(list(contour), num_points)
                    smaller_area_points = [
                        (point[0][0], point[0][1]) 
                        for point in selected_points 
                        if is_point_on_side(point[0], p1, p2, select_left_side)
                    ]
                    random_selected_points = random.sample(
                        smaller_area_points, 
                        min(10, len(smaller_area_points))
                    )
                    sorted_points = sorted(random_selected_points, key=lambda point: point[0])

                    for x, y in sorted_points:
                        saved_coordinates.append({
                            "class_name": f"{cup_class_name}_boundary",
                            "top_point_x": None,
                            "top_point_y": None,
                            "bottom_point_x": None,
                            "bottom_point_y": None,
                            "left_point_x": None,
                            "left_point_y": None,
                            "right_point_x": None,
                            "right_point_y": None,
                            "point_x": int(x),
                            "point_y": int(y),
                            "top_left_x": None,
                            "top_left_y": None,
                            "top_right_x": None,
                            "top_right_y": None,
                            "bottom_left_x": None,
                            "bottom_left_y": None,
                            "bottom_right_x": None,
                            "bottom_right_y": None
                        })

                # カップの特徴点を保存
                print(f"{cup_class_name} の特徴点を保存中...")
                saved_coordinates.append({
                    "class_name": cup_class_name,
                    "top_point_x": int(top_point[0]),
                    "top_point_y": int(top_point[1]),
                    "bottom_point_x": int(bottom_point[0]),
                    "bottom_point_y": int(bottom_point[1]),
                    "left_point_x": int(left_point[0]),
                    "left_point_y": int(left_point[1]),
                    "right_point_x": int(right_point[0]),
                    "right_point_y": int(right_point[1]),
                    "point_x": None,
                    "point_y": None,
                    "top_left_x": None,
                    "top_left_y": None,
                    "top_right_x": None,
                    "top_right_y": None,
                    "bottom_left_x": None,
                    "bottom_left_y": None,
                    "bottom_right_x": None,
                    "bottom_right_y": None
                })

    # 測定の実行
    print("\n測定を実行中...")
    measurements = process_measurements(saved_coordinates)

    # 測定結果をCSVファイルに追加するための新しい行を作成
    measurement_row = {
        "class_name": "measurements",
        "top_point_x": None,
        "top_point_y": None,
        "bottom_point_x": None,
        "bottom_point_y": None,
        "left_point_x": None,
        "left_point_y": None,
        "right_point_x": None,
        "right_point_y": None,
        "point_x": None,
        "point_y": None,
        "top_left_x": None,
        "top_left_y": None,
        "top_right_x": None,
        "top_right_y": None,
        "bottom_left_x": None,
        "bottom_left_y": None,
        "bottom_right_x": None,
        "bottom_right_y": None,
        "right_vertical_line": None,
        "left_vertical_line": None,
    }

    # 測定結果を追加
    for key, value in measurements.items():
        measurement_row[key] = value

    saved_coordinates.append(measurement_row)

    # CSVファイルに保存
    print("\nCSVファイルに結果を保存中...")
    csv_file = "coordinates.csv"
    csv_fields = [
        "class_name", 
        "top_point_x", "top_point_y",
        "bottom_point_x", "bottom_point_y",
        "left_point_x", "left_point_y",
        "right_point_x", "right_point_y",
        "point_x", "point_y",
        "top_left_x", "top_left_y",
        "top_right_x", "top_right_y",
        "bottom_left_x", "bottom_left_y",
        "bottom_right_x", "bottom_right_y",
        "right_anteversion_angle",
        "left_anteversion_angle",
        "right_abduction_angle",
        "left_abduction_angle",
        "right_leg_length",
        "left_leg_length",
        "right_vertical_line",
        "left_vertical_line",
        "obturator_line_length"
    ]

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(saved_coordinates)

    print(f"座標と測定結果が'{csv_file}'に保存されました")

        # 可視化画像を生成して返す
    print("\n可視化画像を生成中...")
    annotated_bgr = visualize_measurements(image_bgr, saved_coordinates, measurements)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return measurements, annotated_rgb



# ========== ここから カメラ起動(Streamlit)（Webアプリ）==========


import streamlit as st

st.set_page_config(page_title="股関節計測", layout="centered")
st.title("股関節計測アプリ")
st.caption("写真を撮る（またはアップロード）すると、前方開角・外転角・脚長を計測します。")

@st.cache_resource
def load_model():
    model_path = "model.pt"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=16EZt6ck39bNJfiHWsTJtlBwppAvDqfgZ"
        gdown.download(url, model_path, quiet=False)
    return YOLO(model_path)

model = load_model()

tab_cam, tab_upload = st.tabs(["カメラで撮影", "画像をアップロード"])
with tab_cam:
    cam_file = st.camera_input("カメラで撮影")
with tab_upload:
    up_file = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])

image_file = cam_file or up_file

if image_file is not None:
    pil_image = Image.open(image_file)
    with st.spinner("計測中..."):
        measurements, annotated_rgb = run_measurement(model, pil_image)

    st.image(annotated_rgb, caption="計測結果", use_container_width=True)

    st.subheader("計測結果")
    col1, col2 = st.columns(2)
    labels = [
        ("右 前方開角", "right_anteversion_angle", "°"),
        ("左 前方開角", "left_anteversion_angle", "°"),
        ("右 外転角", "right_abduction_angle", "°"),
        ("左 外転角", "left_abduction_angle", "°"),
        ("右 脚長", "right_leg_length", "mm"),
        ("左 脚長", "left_leg_length", "mm"),
    ]
    for i, (jp, key, unit) in enumerate(labels):
        col = col1 if i % 2 == 0 else col2
        val = measurements.get(key)
        col.metric(jp, f"{val:.1f} {unit}" if val is not None else "測定不可")
else:
    st.info("カメラで撮影するか、画像をアップロードしてください。")