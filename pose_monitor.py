#coding: UTF-8
import argparse
import logging
import cv2
import sys

import datetime
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import slackweb

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# pandasで移動平均を返す
def getMovingAvg(ndArray):
  if len(ndArray)>0:
    df = pd.DataFrame(ndArray)
    return df.rolling(window=2, min_periods=1).mean()
  else:
    return False

def toNpArray(df):
  return df[0].to_numpy()

# 姿勢が正しいか判定する
# df -> pandasで求めた移動平均群
# threshold 閾値。
def checkPosture(df, threshold):
  max_val = max(df)
  min_val = min(df)
  # 基準値
  standard_val = df[2]

  # 座標の移動量を計算
  moving_up = abs(max_val - standard_val)
  moving_down = abs(standard_val - min_val)

  if moving_up > threshold or moving_down > threshold:
    return True
  else :
    return False

def findPoint(pose, p):
  for point in pose:
    try:
      body_part = point.body_parts[p]
      parts = [0,0]
      # 座標を整数に置換する。切り上げにするため0.5足す
      parts[0] = int(body_part.x * width + 0.5)
      parts[1] = int(body_part.y * height + 0.5)
      return parts
    except:
      return []
  return []

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='姿勢推定するよ')
  parser.add_argument('--camera', type=str, default=1)
  parser.add_argument('--resize', type=str, default='432x368')
  parser.add_argument('--resize-out-ratio', type=float, default=4.0)
  parser.add_argument('--model', type=str, default='cmu')
  args = parser.parse_args()

  w, h = model_wh(args.resize)
  if w > 0 and h > 0:
      e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
  else:
      e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

  # 使用するカメラを指定
  cam = cv2.VideoCapture(args.camera)
  ret_val, image = cam.read()

  global height,width
  initial_time = datetime.datetime.now()
  time_cnt = 0      # 初期値
  execution_cnt = 0 # 初期値
  calclated_cnt = 0 # 初期値
  time_interval = 10 # 定期実行の時間間隔
  threshold_x = 200 # x軸の閾値
  threshold_y = 100 # y軸の閾値
  good_posture = True # いい姿勢かどうか

  # slack通知用の設定
  slack = slackweb.Slack(url="webhookのURL")
  notify_text="姿勢を正しましょう"



  # 部位ごとにnumpy配列で初期化
  lEarX_ndarray = []
  lEarY_ndarray = []
  rEarX_ndarray = []
  rEarY_ndarray = []

  lEyeX_ndarray = []
  lEyeY_ndarray = []
  rEyeX_ndarray = []
  rEyeY_ndarray = []

  noseX_ndarray = []
  noseY_ndarray = []

  while True:

    # 現在時間 sec
    current_time = (datetime.datetime.now() - initial_time).total_seconds()

    # time_interval毎に実行する
    if current_time >= time_cnt :

      ret_val, image = cam.read()
      humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
      pose = humans
      height,width = image.shape[0],image.shape[1]
      # height: 720
      # width: 1280

      if len(pose)>0:
        # 各地点の座標
        left_ear = findPoint(pose, 17)
        right_ear = findPoint(pose, 16)

        left_eye = findPoint(pose, 15)
        right_eye = findPoint(pose, 14)

        nose = findPoint(pose, 0)

        coordinates_info = ''
        # 座標をもとに整形する
        # ================================================================================================= #

        # 左耳
        # ------------------------------------------------------------------------------------------------- #
        if len(left_ear)>0:
          lEarX_ndarray.append(left_ear[0])
          lEarY_ndarray.append(left_ear[1])

          # coordinates_info = coordinates_info + '左耳 (' + str(left_ear[0]) + ',' + str(left_ear[1])+ ') ;'
        # ------------------------------------------------------------------------------------------------- #

        # 右耳
        # ------------------------------------------------------------------------------------------------- #
        if len(right_ear)>0:
          # numpy配列の末尾に追加
          rEarX_ndarray.append(right_ear[0])
          rEarY_ndarray.append(right_ear[1])

          # coordinates_info = coordinates_info + '右耳 (' + str(right_ear[0])+ ',' + str(right_ear[1])+ ') ;'
        # ------------------------------------------------------------------------------------------------- #

        # 右目
        # ------------------------------------------------------------------------------------------------- #
        if len(left_eye)>0:
          lEyeX_ndarray.append(left_eye[0])
          lEyeY_ndarray.append(left_eye[1])

          # coordinates_info = coordinates_info + '左目 (' + str(left_eye[0])+ ',' + str(left_eye[1])+ ') ;'
        # ------------------------------------------------------------------------------------------------- #

        # 右目
        # ------------------------------------------------------------------------------------------------- #
        if len(right_eye)>0:
          rEyeX_ndarray.append(right_eye[0])
          rEyeY_ndarray.append(right_eye[1])

          # coordinates_info = coordinates_info + '右目 (' + str(right_eye[0])+ ',' + str(right_eye[1])+ ') ;'
        # ------------------------------------------------------------------------------------------------- #

        # 鼻
        # ------------------------------------------------------------------------------------------------- #
        if len(nose)>0:
          noseX_ndarray.append(nose[0])
          noseY_ndarray.append(nose[1])

          # coordinates_info = coordinates_info + '鼻 (' + str(nose[0])+ ',' + str(nose[1])+ ') ;'
        # ------------------------------------------------------------------------------------------------- #

        # ログ
        # print(coordinates_info)
        # ================================================================================================= #

        # 次回、定期実行する時刻 time_cntを更新
        time_cnt += time_interval
        execution_cnt += 1

        if execution_cnt > 12 :
          # np.Arrayにcastする
          # =============================================== #
          lEarX_ndarray = np.array(lEarX_ndarray)
          lEarY_ndarray = np.array(lEarY_ndarray)
          lEarX_ndarray = np.array(rEarX_ndarray)
          lEarY_ndarray = np.array(rEarY_ndarray)

          lEyeX_mean = np.array(lEyeX_ndarray)
          lEyeY_mean = np.array(lEyeY_ndarray)
          rEyeX_mean = np.array(rEyeX_ndarray)
          rEyeY_mean = np.array(rEyeY_ndarray)

          noseX_mean = np.array(noseX_ndarray)
          noseY_mean = np.array(noseY_ndarray)
          # =============================================== #

          # 移動平均を求める
          # =============================================== #
          # 左耳の移動平均
          lEarX_mean = getMovingAvg(lEarX_ndarray)
          lEarY_mean = getMovingAvg(lEarY_ndarray)

          # 右耳の移動平均
          rEarX_mean = getMovingAvg(rEarX_ndarray)
          rEarY_mean = getMovingAvg(rEarY_ndarray)

          # 左目の移動平均
          lEyeX_mean = getMovingAvg(lEyeX_ndarray)
          lEyeY_mean = getMovingAvg(lEyeY_ndarray)

          # 右目の移動平均
          rEyeX_mean = getMovingAvg(rEyeX_ndarray)
          rEyeY_mean = getMovingAvg(rEyeY_ndarray)

          # 鼻の移動平均
          noseX_mean = getMovingAvg(noseX_ndarray)
          noseY_mean = getMovingAvg(noseY_ndarray)
          # =============================================== #

          # グラフに描画
          # x座標 → ●、y座標 → ▼
          #
          # 基準となるデータを定義
          # 警告を促した後のはずなので、二週目以降も同じindexの姿勢を正とする
          #
          # =============================================== #
          # 耳の描画
          # 左耳赤、右耳マゼンタ
          # ----------------------------------------------- #
          # if len(lEarX_mean[0])>0 and len(lEarY_mean[0])>0:
          #   plt.plot(lEarX_mean, marker = "o", color = "r")
          #   plt.plot(lEarY_mean, marker = "v", color = "r")

          # if len(rEarX_mean[0])>0 and len(rEarY_mean[0])>0:
          #   plt.plot(rEarX_mean, marker = "o", color = "m")
          #   plt.plot(rEarY_mean, marker = "v", color = "m")
          # ----------------------------------------------- #

          # 目の描画
          # 左目青、右目緑
          # ----------------------------------------------- #
          # if len(lEyeX_mean[0])>0 and len(lEyeY_mean[0])>0:
            # plt.plot(lEyeX_mean, marker = "o", color = "b")
            # plt.plot(lEyeY_mean, marker = "v", color = "b")

          # if len(rEyeX_mean[0])>0 and len(rEyeY_mean[0])>0:
          #   plt.plot(rEyeX_mean, marker = "o", color = "g")
          #   plt.plot(rEyeY_mean, marker = "v", color = "g")
          # ----------------------------------------------- #

          # 鼻の描画
          # 黄色
          # ----------------------------------------------- #
          # if len(noseX_mean[0])>0 and len(noseY_mean[0])>0:
          #   plt.plot(noseX_mean, marker = "o", color = "y")
          #   plt.plot(noseY_mean, marker = "v", color = "y")

          # ----------------------------------------------- #

          # グラフ保存
          # plt.savefig("plt.jpg")

          # 各座標でエラーがあるかを判断する
          # ==================================================================================================================== #
          # 目のエラー判定
          # ------------------------------------------------------------------------------------- #
          # x座標エラーがあるかどうか
          if checkPosture(rEyeX_mean[0], threshold_x) or checkPosture(lEyeX_mean[0], threshold_x):
            good_posture = False
          else :
            # y座標エラーがあるかどうか
            if checkPosture(rEyeY_mean[0], threshold_y) or checkPosture(lEyeY_mean[0], threshold_y):
              good_posture = False
            else :
              good_posture = True

          # ------------------------------------------------------------------------------------- #

          # 耳のエラー判定
          # ------------------------------------------------------------------------------------- #
          # x座標エラーがあるかどうか
          if checkPosture(rEarX_mean[0], threshold_x) or checkPosture(lEarX_mean[0], threshold_x):
            good_posture = False
          else :
            # y座標エラーがあるかどうか
            if checkPosture(rEarY_mean[0], threshold_y) or checkPosture(lEarY_mean[0], threshold_y):
              good_posture = False
            else :
              good_posture = True
          # ------------------------------------------------------------------------------------- #

          # 鼻のエラー判定
          # ------------------------------------------------------------------------------------- #
          # x座標エラーがあるかどうか
          if checkPosture(noseX_mean[0], threshold_x):
            good_posture = False
          else:
            # y座標エラーがあるかどうか
            if checkPosture(noseY_mean[0], threshold_y):
              good_posture = False
            else :
              good_posture = True

          # ------------------------------------------------------------------------------------- #
          # ==================================================================================================================== #

          # 姿勢が悪かったら通知を飛ばす
          if good_posture == 0:
            slack.notify(text=notify_text)

          # 完了し次第初期化
          execution_cnt = 0
          lEarX_ndarray = []
          lEarY_ndarray = []
          rEarX_ndarray = []
          rEarY_ndarray = []

          lEyeX_ndarray = []
          lEyeY_ndarray = []
          rEyeX_ndarray = []
          rEyeY_ndarray = []

          noseX_ndarray = []
          noseY_ndarray = []

  cv2.destroyAllWindows()









