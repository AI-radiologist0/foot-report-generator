"""

    reference from https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/pose_estimation/_init_paths.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# 현재 파일(_init_path.py)의 절대 경로 찾기
this_dir = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(this_dir, ".."))

# lib 디렉토리 경로 추가
lib_path = os.path.join(project_root, "lib")

if lib_path not in sys.path:
    sys.path.append(lib_path)