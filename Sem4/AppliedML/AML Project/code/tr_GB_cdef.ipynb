{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZCahqrluJbR3"
   },
   "source": [
    "#### Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install xgboost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "executionInfo": {
     "elapsed": 3857,
     "status": "ok",
     "timestamp": 1746045475376,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "AgcpvrsjIfsu"
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import time\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "\n",
    "from gensim.models import Word2Vec\n",
    "\n",
    "import xgboost as xgb\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DeIjNjA7JhS-"
   },
   "source": [
    "#### Preprocessing and Splitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "executionInfo": {
     "elapsed": 50010,
     "status": "ok",
     "timestamp": 1746045612608,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "-8YcZRmUIhQS"
   },
   "outputs": [],
   "source": [
    "# read/prep data\n",
    "dat = pd.read_csv(\"../data/tokenized_reviews.csv\")\n",
    "dat = dat.dropna()\n",
    "dat[\"quote\"] = dat[\"quote\"].astype(int)\n",
    "dat[\"tokenized_words\"] = dat[\"tokenized_words\"].apply(lambda x: x.strip(\"[']\").replace(\"', '\",\" \"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "executionInfo": {
     "elapsed": 2086,
     "status": "ok",
     "timestamp": 1746045614692,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "odi_TJXNIk8r"
   },
   "outputs": [],
   "source": [
    "# 85% train / 15% test\n",
    "X_train, X_test, y_train, y_test = train_test_split(dat.drop(columns=[\"popular\"]),\n",
    "                                                    dat[\"popular\"],\n",
    "                                                    test_size = 0.15,\n",
    "                                                    random_state = 229,\n",
    "                                                    stratify = dat[\"popular\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1453600, 13), (1453600,))"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, y_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5OTHg7coV5pL"
   },
   "source": [
    "#### Down Sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "executionInfo": {
     "elapsed": 544,
     "status": "ok",
     "timestamp": 1746045615261,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "0l3akNYRIm5k"
   },
   "outputs": [],
   "source": [
    "# undersample train set\n",
    "majority_size = len(y_train[y_train==0])\n",
    "minority_size = len(y_train[y_train==1])\n",
    "majority_indices = y_train[y_train==0].index\n",
    "rng = np.random.default_rng(seed=229)\n",
    "drop_indices = rng.choice(majority_indices, majority_size-minority_size, replace=False)\n",
    "X_train = X_train.drop(drop_indices)\n",
    "y_train = y_train.drop(drop_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((431298, 13), (431298,))"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical_cols = [col for col in X_train.columns if col != 'tokenized_words']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "R1Bb8VUPJuzX"
   },
   "source": [
    "#### BOW (Undersampling)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FxPvvWOGJw1V"
   },
   "source": [
    "#### TF-IDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "background_save": true,
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8880163,
     "status": "ok",
     "timestamp": 1746067709741,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "QGd_9g1-JUy9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "XGBOOST - RANDOM FOREST - TF-IDF\n",
      "Fitting 1 folds for each of 8 candidates, totalling 8 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x11001dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107511d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x112019d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104029d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.249 total time= 5.8min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106669d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.235 total time= 6.4min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10abd1d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.255 total time=11.5min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104fadd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.245 total time=13.1min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x102bb5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=1000;, score=0.269 total time=18.3min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105349d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=1000;, score=0.264 total time=19.0min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106a29d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=1000;, score=0.263 total time=29.5min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10651dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=1000;, score=0.270 total time=32.3min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10505dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: #000;\n",
       "  --sklearn-color-text-muted: #666;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: flex;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "  align-items: start;\n",
       "  justify-content: space-between;\n",
       "  gap: 0.5em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label .caption {\n",
       "  font-size: 0.6rem;\n",
       "  font-weight: lighter;\n",
       "  color: var(--sklearn-color-text-muted);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 0.5em;\n",
       "  text-align: center;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;tfidfvectorizer&#x27;,\n",
       "                                                                         TfidfVectorizer(),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,\n",
       "                                                                          &#x27;rating_diff&#x27;,\n",
       "                                                                          &#x27;nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 1000)},\n",
       "             verbose=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>GridSearchCV</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html\">?<span>Documentation for GridSearchCV</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;tfidfvectorizer&#x27;,\n",
       "                                                                         TfidfVectorizer(),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,\n",
       "                                                                          &#x27;rating_diff&#x27;,\n",
       "                                                                          &#x27;nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 1000)},\n",
       "             verbose=3)</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>best_estimator_: Pipeline</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                   transformers=[(&#x27;tfidfvectorizer&#x27;,\n",
       "                                                  TfidfVectorizer(),\n",
       "                                                  &#x27;tokenized_words&#x27;),\n",
       "                                                 (&#x27;standardscaler&#x27;,\n",
       "                                                  StandardScaler(),\n",
       "                                                  [&#x27;user_reviews&#x27;,\n",
       "                                                   &#x27;days_since_review&#x27;,\n",
       "                                                   &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;,\n",
       "                                                   &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;,\n",
       "                                                   &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                                   &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;,\n",
       "                                                   &#x27;quote&#x27;, &#x27;sentiment&#x27;])])),\n",
       "                (...\n",
       "                              feature_types=None, feature_weights=None,\n",
       "                              gamma=None, grow_policy=None,\n",
       "                              importance_type=None,\n",
       "                              interaction_constraints=None, learning_rate=0.1,\n",
       "                              max_bin=None, max_cat_threshold=None,\n",
       "                              max_cat_to_onehot=None, max_delta_step=None,\n",
       "                              max_depth=6, max_leaves=None,\n",
       "                              min_child_weight=None, missing=nan,\n",
       "                              monotone_constraints=None, multi_strategy=None,\n",
       "                              n_estimators=1000, n_jobs=-1,\n",
       "                              num_parallel_tree=None, ...))])</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>columntransformer: ColumnTransformer</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html\">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                  transformers=[(&#x27;tfidfvectorizer&#x27;, TfidfVectorizer(),\n",
       "                                 &#x27;tokenized_words&#x27;),\n",
       "                                (&#x27;standardscaler&#x27;, StandardScaler(),\n",
       "                                 [&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;,\n",
       "                                  &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;,\n",
       "                                  &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                  &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;,\n",
       "                                  &#x27;sentiment&#x27;])])</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>tfidfvectorizer</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>tokenized_words</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>TfidfVectorizer</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\">?<span>Documentation for TfidfVectorizer</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>TfidfVectorizer()</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>standardscaler</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;, &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;, &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;, &#x27;sentiment&#x27;]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>StandardScaler</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html\">?<span>Documentation for StandardScaler</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>remainder</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>passthrough</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>XGBRegressor</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.XGBRegressor\">?<span>Documentation for XGBRegressor</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=&#x27;error&#x27;, feature_types=None,\n",
       "             feature_weights=None, gamma=None, grow_policy=None,\n",
       "             importance_type=None, interaction_constraints=None,\n",
       "             learning_rate=0.1, max_bin=None, max_cat_threshold=None,\n",
       "             max_cat_to_onehot=None, max_delta_step=None, max_depth=6,\n",
       "             max_leaves=None, min_child_weight=None, missing=nan,\n",
       "             monotone_constraints=None, multi_strategy=None, n_estimators=1000,\n",
       "             n_jobs=-1, num_parallel_tree=None, ...)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[('columntransformer',\n",
       "                                        ColumnTransformer(remainder='passthrough',\n",
       "                                                          transformers=[('tfidfvectorizer',\n",
       "                                                                         TfidfVectorizer(),\n",
       "                                                                         'tokenized_words'),\n",
       "                                                                        ('standardscaler',\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         ['user_reviews',\n",
       "                                                                          'days_since_review',\n",
       "                                                                          'user_rating',\n",
       "                                                                          'rating_diff',\n",
       "                                                                          'nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'xgbregressor__learning_rate': (0.1, 0.3),\n",
       "                         'xgbregressor__max_depth': (4, 6),\n",
       "                         'xgbregressor__n_estimators': (100, 1000)},\n",
       "             verbose=3)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# TF-IDF\n",
    "print(\"\\n\\nXGBOOST - RANDOM FOREST - TF-IDF\")\n",
    "\n",
    "# pipeline\n",
    "tfidf_pipe = make_pipeline(\n",
    "    ColumnTransformer(remainder='passthrough',\n",
    "                      transformers=[('tfidfvectorizer',TfidfVectorizer(),'tokenized_words'),\n",
    "                                   ('standardscaler', StandardScaler(), numerical_cols) ]),\n",
    "    xgb.XGBRegressor(objective='binary:logistic',\n",
    "                     eval_metric='error',\n",
    "                     seed=229,\n",
    "                     n_jobs=-1))\n",
    "\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'xgbregressor__n_estimators': (100,1000),\n",
    "    'xgbregressor__max_depth': (4,6),\n",
    "    'xgbregressor__learning_rate': (0.1, 0.3)\n",
    "}\n",
    "\n",
    "# perform validation\n",
    "gs_tfidf_pipe = GridSearchCV(tfidf_pipe,\n",
    "                           parameters,\n",
    "                           cv=ShuffleSplit(n_splits=1,\n",
    "                                           test_size=0.15,\n",
    "                                           random_state=229), n_jobs=-1,verbose=3)\n",
    "gs_tfidf_pipe.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "background_save": true,
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 2,
     "status": "ok",
     "timestamp": 1746067709762,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "kvVYELMbui0F"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([ 373.97754383, 1134.23915291,  780.62015629, 1930.76124287,\n",
      "        339.48315334, 1093.31210279,  680.69858193, 1766.66438985]), 'std_fit_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'mean_score_time': array([8.65802813, 5.55718708, 5.32359695, 5.40668583, 8.99425673,\n",
      "       5.09616423, 7.42966819, 5.02466726]), 'std_score_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'param_xgbregressor__learning_rate': masked_array(data=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=1e+20), 'param_xgbregressor__max_depth': masked_array(data=[4, 4, 6, 6, 4, 4, 6, 6],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'param_xgbregressor__n_estimators': masked_array(data=[100, 1000, 100, 1000, 100, 1000, 100, 1000],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'params': [{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}], 'split0_test_score': array([0.23509264, 0.26435727, 0.24509281, 0.27009428, 0.24866152,\n",
      "       0.26889634, 0.25458896, 0.26298058]), 'mean_test_score': array([0.23509264, 0.26435727, 0.24509281, 0.27009428, 0.24866152,\n",
      "       0.26889634, 0.25458896, 0.26298058]), 'std_test_score': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'rank_test_score': array([8, 3, 7, 1, 6, 2, 5, 4], dtype=int32)}\n",
      "{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}\n",
      "\n",
      "Best model saved as 'xgboost_tfidf_model_cde.pkl'\n"
     ]
    }
   ],
   "source": [
    "# print save\n",
    "print(gs_tfidf_pipe.cv_results_)\n",
    "print(gs_tfidf_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./xgboost_tfidf_model_cde.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_tfidf_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'xgboost_tfidf_model_cde.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "background_save": true
    },
    "id": "vft53tvnJX-w"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Top 10 features:\n",
      "['user_reviews' 'num_words' 'ftc' 'katniss' 'fuck' 'maas' 'fucking'\n",
      " 'cinder' 'thanks' 'quote']\n"
     ]
    }
   ],
   "source": [
    "sorted_ind = gs_tfidf_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_.argsort()[::-1]\n",
    "\n",
    "tfidf_feature_names = gs_tfidf_pipe.best_estimator_.named_steps['columntransformer'].transformers_[0][1].get_feature_names_out()\n",
    "num_tfidf_features = len(tfidf_feature_names)\n",
    "\n",
    "passthrough_columns = X_train.drop(columns=['tokenized_words']).columns.tolist()\n",
    "all_feature_names = list(tfidf_feature_names) + passthrough_columns\n",
    "\n",
    "# Print the top 10 feature names and their importances\n",
    "top_50_indices = sorted_ind[:10]\n",
    "print(\"Top 10 features:\")\n",
    "print(np.array(all_feature_names)[top_50_indices])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "background_save": true
    },
    "id": "UN-BU2xFbizF"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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GX/rrbfM3Gul1yvXRc506EjmSnd+aLCOi4szIwx0OHmFChXkgQcekFdffVVuv/12mTBhgtmmc8WXLFliFmXz9WVYOAAAAAB4XegeNGiQNS2xGQ3WerkwrXArDcV6WbD//ve/Uq9evQs61rhx48x51+frKuVHjx41QVznhOuK5TfccINZfbx69erSrVs38fG54CvBAQAAAAAs4HByvS/ksCCArpqu1/zWy5gBdqKjTXT19+7duzNnCrZD/4ad0b9hZ/Rv785Nus5ZcHBwrvtREgUAAAAAwCKEbgAAAAAALELoBgAAAADAIoRuAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALOJn1YHh/VrHrJE0vyBPNwPZxM/o4ekmAAAAAMgnKt0AAAAAAFiE0J1PgwcPlt69e7t/79Spk0RGRlr6muvXrxeHwyGnTp2y9HUAAAAAANYgdBcROYX4tm3byq+//iply5b1WLsAAAAAABePOd1FWIkSJaRKlSqebgYAAAAA4CIVq9CdkZEhs2bNkldeeUV+/PFHqVy5stx7770yadIk2bt3r4wePVq2bt0qgYGBcuutt8ozzzwjpUuXztexk5OTzXHefvttMxy8YcOGMnPmTFPBdtm8ebPZ56uvvpKAgABp1aqVLFmyRMaMGSMbNmwwt3nz5pl9jx49KvHx8XL99dfLn3/+KeXKlTPb33//fZk6daocPnxYQkND5cEHH5SHHnrI/Ro1atSQESNGmMffe+89KV++vEyePNlsO1/b9eaSmJho7gN8nOLr67yIMw0rpaameroJtjh/nEfYEf0bdkb/hp3Rv71Tfj+vYhW6J06cKK+++qrMmTNH2rdvb4Zuf/fdd/LXX39J165dpU2bNrJt2zY5ceKEDBs2TEaNGiULFizI17F13/3795sQXbVqVVm2bJl069bNhPk6derI7t27JTw8XO655x4TrP38/GTdunWSnp5ufj948KAJ6tOmTTPHq1Spkgndme3YsUP69esnUVFR0r9/f9myZYvcf//9EhISYuacu8yePVsef/xxefTRR2Xp0qVy3333SceOHaVevXo5tj0mJkaio6PP2T65WYYEBqZf4FmG1ZYvX+7pJthCbGysp5sAWIb+DTujf8PO6N/eJSkpKV/7OZxOZ7EoZZ4+fdoE2eeff94E6sw0iD/8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y36CtKqsATY+Pl569uxpAvX7778vO3bskBdeeOGchdFKlSrlnoOdWeZQrHTous7h7t69u3z66aeya9cumTRp0iUvsnah7wsAAAAAUPiKbaU7NxqyNaTqQmQ6LFu9++67F328LVu2SPXq1U3Qdvnhhx+y7KNVcx3ufj7169c3w9p1brcr2G/evNm0sV69ehfdPgAAAACAdYptpTs3tWvXNnMqnnvuObOI2htvvCEvv/zyRR+vTp06cuzYMTOnWoeX6zDz7HO+dS740aNHzZxunTeeeX61i1bLdbX1QYMGmYXXdKG0Bx98UO666y730HIAAAAAQNFC6M5G503rZbh0ZfKGDRvKW2+9ZeZ3X6ybbrpJxowZY1YZ11XEtfI9ZcqULPvopb+6desm119/vVSqVEnefvvtc46jc8lXrVolf/zxh7Rs2VL69u0r4eHh8vzzz1902wAAAAAA1iqWq5fj/HTxOL1smVbd9XJogJ3oSBZd+V/XWWChEtgN/Rt2Rv+GndG/vTs35bV6OZVuAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALELoBgAAAADAIoRuAAAAAAAs4mfVgeH9WseskTS/IE83w/biZ/TwdBMAAAAAWIRKNwAAAAAAFiF0AwAAAABgEUI3AAAAAAAWIXQDAAAAAGARQreHdOrUSR588EGJjIyU8uXLS+XKleXVV1+Vv/76S4YMGSJlypSR2rVry4oVK8z+6enpMnToUKlZs6aUKlVK6tWrJ/PmzctyzMGDB0vv3r1l1qxZEhoaKiEhIfLAAw9Iamqqh94lAAAAABRvrF7uQQsXLpQJEybIV199Je+8847cd999smzZMunTp488+uijMmfOHLnrrrvk2LFj4u/vL1dccYW89957Jkxv2bJFRowYYcJ1v3793Mdct26d2ab3hw8flv79+0vTpk1l+PDhubYjOTnZ3FwSExPNfYCPU3x9nRafBfCliGfON+cddkT/hp3Rv2Fn9G/vlN/Py+F0OklVHqp0a/V648aN5nf9uWzZsnLLLbfIokWLzLbjx4+bAL1161a59tprzznGqFGjzD5Lly51V7rXr18vR44cEV9fX7NNA7mPj48sWbIk17ZERUVJdHT0OdsXL14sgYGBBfaeAQAAAMAukpKSZODAgZKQkCDBwcG57kel24MaN27s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",
      "text/plain": [
       "<Figure size 1000x1200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot feature importance\n",
    "fig, ax = plt.subplots(figsize=(10, 12))\n",
    "xgb.plot_importance(\n",
    "    gs_tfidf_pipe.best_estimator_.named_steps['xgbregressor'],\n",
    "    max_num_features=50,\n",
    "    ax=ax,\n",
    "    importance_type='gain',\n",
    "    show_values=False,\n",
    "    height=0.6\n",
    ")\n",
    "\n",
    "tick_labels = [item.get_text() for item in ax.get_yticklabels()]\n",
    "for i, label in enumerate(tick_labels):\n",
    "    try:\n",
    "        feature_index = int(label[1:])  # label is like 'f123'\n",
    "        if 0 <= feature_index < len(all_feature_names):\n",
    "            tick_labels[i] = all_feature_names[feature_index]\n",
    "    except ValueError:\n",
    "        pass\n",
    "ax.set_yticklabels(tick_labels)\n",
    "\n",
    "plt.title(\"Feature Importance - XGBoost (TF-IDF)\", fontsize=16)\n",
    "plt.xlabel(\"F score\", fontsize=12)\n",
    "plt.ylabel(\"Features\", fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"rf_tfidf.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "background_save": true
    },
    "id": "gNauYTXWDabz"
   },
   "outputs": [],
   "source": [
    "# predict\n",
    "predictions = gs_tfidf_pipe.predict(X_test)\n",
    "predictions = list(map(round,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[157290  61172]\n",
      " [ 10451  27605]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.72      0.81    218462\n",
      "           1       0.31      0.73      0.44     38056\n",
      "\n",
      "    accuracy                           0.72    256518\n",
      "   macro avg       0.62      0.72      0.62    256518\n",
      "weighted avg       0.84      0.72      0.76    256518\n",
      "\n",
      "Specificity : 0.7199879155184883\n",
      "ROC-AUC : 0.722683152629961\n"
     ]
    }
   ],
   "source": [
    "# evaluate\n",
    "cm = confusion_matrix(y_test, predictions)\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()\n",
    "print(confusion_matrix(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Specificity :\", tn/(fp+tn))\n",
    "print(\"ROC-AUC :\", roc_auc_score(y_test, predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "N1tKguZhuOQ1"
   },
   "source": [
    "### CBoW"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 609
    },
    "executionInfo": {
     "elapsed": 1749522,
     "status": "ok",
     "timestamp": 1746042240960,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "rcTvR_rqJL9U",
    "outputId": "ec0da104-8477-4ad9-d5bf-0108af9d3ed0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "XGBOOST - RANDOM FOREST - CBoW\n",
      "Fitting 1 folds for each of 8 candidates, totalling 8 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106b79d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104a15d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10696dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105d1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: #000;\n",
       "  --sklearn-color-text-muted: #666;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-2 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-2 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-2 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: flex;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "  align-items: start;\n",
       "  justify-content: space-between;\n",
       "  gap: 0.5em;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label .caption {\n",
       "  font-size: 0.6rem;\n",
       "  font-weight: lighter;\n",
       "  color: var(--sklearn-color-text-muted);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-2 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-2 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-2 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-2 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 0.5em;\n",
       "  text-align: center;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-2 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-2 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;countvectorizer&#x27;,\n",
       "                                                                         CountVectorizer(),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,\n",
       "                                                                          &#x27;rating_diff&#x27;,\n",
       "                                                                          &#x27;nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 1000)},\n",
       "             verbose=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" ><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>GridSearchCV</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html\">?<span>Documentation for GridSearchCV</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;countvectorizer&#x27;,\n",
       "                                                                         CountVectorizer(),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,\n",
       "                                                                          &#x27;rating_diff&#x27;,\n",
       "                                                                          &#x27;nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 1000)},\n",
       "             verbose=3)</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" ><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>best_estimator_: Pipeline</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                   transformers=[(&#x27;countvectorizer&#x27;,\n",
       "                                                  CountVectorizer(),\n",
       "                                                  &#x27;tokenized_words&#x27;),\n",
       "                                                 (&#x27;standardscaler&#x27;,\n",
       "                                                  StandardScaler(),\n",
       "                                                  [&#x27;user_reviews&#x27;,\n",
       "                                                   &#x27;days_since_review&#x27;,\n",
       "                                                   &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;,\n",
       "                                                   &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;,\n",
       "                                                   &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                                   &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;,\n",
       "                                                   &#x27;quote&#x27;, &#x27;sentiment&#x27;])])),\n",
       "                (...\n",
       "                              feature_types=None, feature_weights=None,\n",
       "                              gamma=None, grow_policy=None,\n",
       "                              importance_type=None,\n",
       "                              interaction_constraints=None, learning_rate=0.1,\n",
       "                              max_bin=None, max_cat_threshold=None,\n",
       "                              max_cat_to_onehot=None, max_delta_step=None,\n",
       "                              max_depth=6, max_leaves=None,\n",
       "                              min_child_weight=None, missing=nan,\n",
       "                              monotone_constraints=None, multi_strategy=None,\n",
       "                              n_estimators=1000, n_jobs=-1,\n",
       "                              num_parallel_tree=None, ...))])</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" ><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>columntransformer: ColumnTransformer</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html\">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                  transformers=[(&#x27;countvectorizer&#x27;, CountVectorizer(),\n",
       "                                 &#x27;tokenized_words&#x27;),\n",
       "                                (&#x27;standardscaler&#x27;, StandardScaler(),\n",
       "                                 [&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;,\n",
       "                                  &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;,\n",
       "                                  &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                  &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;,\n",
       "                                  &#x27;sentiment&#x27;])])</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" ><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>countvectorizer</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>tokenized_words</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" ><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>CountVectorizer</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html\">?<span>Documentation for CountVectorizer</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>CountVectorizer()</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-16\" type=\"checkbox\" ><label for=\"sk-estimator-id-16\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>standardscaler</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;, &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;, &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;, &#x27;sentiment&#x27;]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-17\" type=\"checkbox\" ><label for=\"sk-estimator-id-17\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>StandardScaler</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html\">?<span>Documentation for StandardScaler</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-18\" type=\"checkbox\" ><label for=\"sk-estimator-id-18\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>remainder</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-19\" type=\"checkbox\" ><label for=\"sk-estimator-id-19\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>passthrough</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-20\" type=\"checkbox\" ><label for=\"sk-estimator-id-20\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>XGBRegressor</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.XGBRegressor\">?<span>Documentation for XGBRegressor</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=&#x27;error&#x27;, feature_types=None,\n",
       "             feature_weights=None, gamma=None, grow_policy=None,\n",
       "             importance_type=None, interaction_constraints=None,\n",
       "             learning_rate=0.1, max_bin=None, max_cat_threshold=None,\n",
       "             max_cat_to_onehot=None, max_delta_step=None, max_depth=6,\n",
       "             max_leaves=None, min_child_weight=None, missing=nan,\n",
       "             monotone_constraints=None, multi_strategy=None, n_estimators=1000,\n",
       "             n_jobs=-1, num_parallel_tree=None, ...)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.15, train_size=None),\n",
       "             estimator=Pipeline(steps=[('columntransformer',\n",
       "                                        ColumnTransformer(remainder='passthrough',\n",
       "                                                          transformers=[('countvectorizer',\n",
       "                                                                         CountVectorizer(),\n",
       "                                                                         'tokenized_words'),\n",
       "                                                                        ('standardscaler',\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         ['user_reviews',\n",
       "                                                                          'days_since_review',\n",
       "                                                                          'user_rating',\n",
       "                                                                          'rating_diff',\n",
       "                                                                          'nu...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'xgbregressor__learning_rate': (0.1, 0.3),\n",
       "                         'xgbregressor__max_depth': (4, 6),\n",
       "                         'xgbregressor__n_estimators': (100, 1000)},\n",
       "             verbose=3)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.235 total time=  39.7s\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1054a5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.248 total time=  38.6s\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1048ddd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.255 total time=  54.7s\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1049c9d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.244 total time= 1.0min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x102a29d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=1000;, score=0.271 total time= 2.2min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x102e1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=1000;, score=0.265 total time= 2.3min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x11081dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=1000;, score=0.271 total time= 3.8min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106d99d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=1000;, score=0.272 total time= 4.1min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106de1d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    }
   ],
   "source": [
    "# BAG OF WORDS\n",
    "print(\"\\n\\nXGBOOST - RANDOM FOREST - CBoW\")\n",
    "\n",
    "# pipeline\n",
    "bow_pipe = make_pipeline(\n",
    "    ColumnTransformer(remainder='passthrough',\n",
    "                      transformers=[('countvectorizer', CountVectorizer(),'tokenized_words'), \n",
    "                                   ('standardscaler', StandardScaler(), numerical_cols)]),\n",
    "    xgb.XGBRegressor(objective='binary:logistic',\n",
    "                     eval_metric='error',\n",
    "                     seed=229,\n",
    "                     n_jobs=-1))\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'xgbregressor__n_estimators': (100,1000),\n",
    "    'xgbregressor__max_depth': (4,6),\n",
    "    'xgbregressor__learning_rate': (0.1, 0.3)\n",
    "}\n",
    "\n",
    "# perform validation\n",
    "gs_bow_pipe = GridSearchCV(bow_pipe,\n",
    "                           parameters,\n",
    "                           cv=ShuffleSplit(n_splits=1,\n",
    "                                           test_size=0.15,\n",
    "                                           random_state=229),n_jobs=-1,verbose=3)\n",
    "gs_bow_pipe.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 12,
     "status": "ok",
     "timestamp": 1746042240986,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "l4RjYMeYahHQ",
    "outputId": "d41071a7-c207-4a6e-cf70-c62c95609bdb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([ 35.84392333, 133.30427289,  56.49859381, 243.32555199,\n",
      "        34.75194192, 127.90338588,  51.00735378, 224.20040798]), 'std_fit_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'mean_score_time': array([3.87634468, 4.35574102, 3.73140812, 4.63173294, 3.82987905,\n",
      "       4.27692604, 3.72735119, 4.52409983]), 'std_score_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'param_xgbregressor__learning_rate': masked_array(data=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=1e+20), 'param_xgbregressor__max_depth': masked_array(data=[4, 4, 6, 6, 4, 4, 6, 6],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'param_xgbregressor__n_estimators': masked_array(data=[100, 1000, 100, 1000, 100, 1000, 100, 1000],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'params': [{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}], 'split0_test_score': array([0.23458618, 0.26460612, 0.24435121, 0.27153009, 0.24817914,\n",
      "       0.27139962, 0.25531596, 0.27078158]), 'mean_test_score': array([0.23458618, 0.26460612, 0.24435121, 0.27153009, 0.24817914,\n",
      "       0.27139962, 0.25531596, 0.27078158]), 'std_test_score': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'rank_test_score': array([8, 4, 7, 1, 6, 2, 5, 3], dtype=int32)}\n",
      "{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}\n",
      "\n",
      "Best model saved as 'xgboost_bow_model_cde.pkl'\n"
     ]
    }
   ],
   "source": [
    "# print model\n",
    "print(gs_bow_pipe.cv_results_)\n",
    "print(gs_bow_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./xgboost_bow_model_cde.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_bow_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'xgboost_bow_model_cde.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 8,
     "status": "ok",
     "timestamp": 1746042288911,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "_0iRQGUVJPvM",
    "outputId": "c43df0ba-b9cd-469d-f362-7eca9492949b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['katniss' 'ftc' 'maas' 'cinder' 'fuck' 'amazon' 'thank' 'thanks' 'hype'\n",
      " 'fucking']\n"
     ]
    }
   ],
   "source": [
    "# feature importance\n",
    "sorted_ind = gs_bow_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_.argsort()[::-1]\n",
    "feature_names = gs_bow_pipe.best_estimator_.named_steps['columntransformer'].transformers_[0][1].get_feature_names_out()\n",
    "num_bow_features = len(feature_names)\n",
    "bow_indices = sorted_ind[sorted_ind < num_bow_features]\n",
    "top_bow_features = np.take(feature_names, bow_indices[:10])\n",
    "\n",
    "# Print the top 10 CountVectorizer features\n",
    "print(top_bow_features)\n",
    "# print(np.take(gs_bow_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_, bow_indices[:50]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 944
    },
    "executionInfo": {
     "elapsed": 951,
     "status": "ok",
     "timestamp": 1746042289868,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "ewT3DeXTSctd",
    "outputId": "9732e13c-217b-4857-ed73-b7157e3ae9b7"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA90AAASlCAYAAAC1GLqkAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs3Qd4FGX38OGTRiD0JkUCSJVepCPSBSkKFlQsoPQqIiBIMRSlCAJ2sdARFQWVIk3gRUA6SBOFlwgqiCgdDCTZ7zrP+83+N8kGQpLJJju/+7rWze7Ozjw7Z4M5c54S4HK5XAIAAAAAAFJdYOrvEgAAAAAAKJJuAAAAAABsQtINAAAAAIBNSLoBAAAAALAJSTcAAAAAADYh6QYAAAAAwCYk3QAAAAAA2ISkGwAAAAAAm5B0AwAAAABgE5JuAPCB4sWLS0BAwA1v06ZN83Uz/UKjRo3M+YyIiPB1U+AnvvvuOwkMDJSsWbPKkSNHEt1u9OjR5rtXoUIFiYqK8rrNL7/8IkOGDJFatWpJ/vz5JSQkRHLmzCnly5eXJ598UhYsWCBXr15N8L5Zs2Z5/XdD31+oUCFp27atLFu2TJysffv2kiVLFvntt9+8/psQ/xYWFiblypWTfv36yfHjx2+4b5fLJZ9++qk8+OCDEh4eLpkzZ5bcuXNL1apVTTwTe/9TTz1ljtW3b99E912qVCmzjd4nRt+v2+j+LOPGjTPPLV++/IZtB5D2SLoBwIfq168vnTp18nrTP7rTWmRkpPmjTS8KwH907tzZxFUTNaRckyZNpHfv3nLlyhVzbmNjYxNss2vXLpMEBQcHy5w5cyQ0NDTO69HR0fLCCy/InXfeKa+99pocPnxYqlSpIo888ojZf6ZMmUzC/cQTT5jfx8SSe038Pf/duP/++03SvnTpUmnTpo0MGzZMMpL169eb76omximxZs0aWbJkiUlOixQp4nUbPd/WeXv66aflnnvukVOnTslbb70llSpVku3bt3t93x9//CF16tSRxx57zByjYMGC0q5dO2nQoIH8/vvvJp5lypSRt99+O8F7Gzdu7P6c3pw4cUKOHj1qftb7+BcMLOvWrYuzP/X8889LgQIFzP3169dveo4ApCEXACDNFStWzKX/BM+cOdOVnhw7dsy0S9vnLxo2bGg+08svv+xyqk6dOqXL71tGdunSJVfJkiXNeZ00aVKc1/79919XhQoVzGujRo3y+v5HH33UvJ4jRw7XRx995Lp+/XqCbU6dOuUaPXq02WbLli1xXtNY3uh39fXXXzevBwQEuH788UdXRrFu3TrTbv29TYmKFSu6MmfO7Dpz5swt/Ztw7tw5V4MGDczrd911V4LX//nnH1eJEiXM69WqVXPt378/zusax8mTJ7uCgoLMNtOnT4/z+n//+1/zvN7+/PPPBPufM2eOea169ermfu7cuQm20fdZ+9D9edJjezsuAN+i0g0AAHCLtMKsPQe0m/nIkSPl4MGD7tf08YEDB6R69eoyYsSIBO/96KOPTNdkrWavXbtWnn32WVMRj0+rlqNGjTL7Klas2C21T6udWuHVbtDaHd5JVq9eLfv37zfV57x5897Se7WXgJ5ztXPnTjl//nyc17Vy/t///lfuuOMOc1516IAnjaP2YJg+fbp5PGjQIDl06JD7dX2fFUtv1W7rOasNVkXb2za6H92fJ63Y6xCDN954w8QeQPpA0g0AGYT+AahdTYsWLWq6qubJk0datGiR6Pg9TQJefvll04X99ttvN3/g6x+gzZo1k88++yzB9tpN1voD7tdff00w3jGpXZWtsaa6XWLP//PPPzJgwAApWbKk+Szxu5JqIqJjJXVsqrb7tttuM+Mzt2zZkqxzd7N26h/WAwcONN14dWxm6dKlZeLEie5uw9pltEePHmbspra3bNmy8uabb3rdrzVeVP8w3rBhg9x7770mVjpeVMftzp07N9E2aZfj9957T+rVq2f++Lfa0r9/f9MGbzzjM3PmTKlbt655rz5nDReYPXu2ef2ZZ56JE1PPce7btm1zjy3W7rJ63jXp07HB2lX3Zufw8uXLpiuzjkPVc6T70G67ibXbOq+DBw82XXmzZ89uElntlqv727x5c4LtdWzzlClTTNfeXLlymfOjsdB2//3335LW7r77bpPc6nht/awaP223tlHPn553TYA8aSKk3c5Vnz59pEaNGjc9jibP+rtwqzR+Stvljca8Q4cOUrhwYffvmcZbk9bU/I7quHW9sKD/vuh3I1u2bCZhbN26tfnOev7uWN2l9XfH87t6K0NetHu4iv9vUFLpd9fz81o02V64cKH5efLkyeY7mBgdfqDd17Wb96RJk5LcxVyf07H9OkxA43KjxNyza7lF39uqVSvTNf3bb79N4icGYDsfV9oBwJFutXv5tGnTXIGBgeY9VatWdT388MOuu+++25UpUybznHZBja9Lly7mtTvvvNPVokUL0521bt267v08//zzcbb/4IMPXA899JB5LWvWrKZLsuctqV2VrW6vnu/xfL5169auO+64w5U7d27X/fff73rkkUdcTzzxhHu7F154wWyn7axVq5Z5vXbt2qabrHbZ/Pjjj123IrGupFZ7HnjgAVe5cuVct912m/n89957rytLlizmtb59+7qOHDniKliwoCs8PNzVoUMHV+PGjd1dRydMmJDo8fr3728+Q/ny5V2PPfaY65577nGf+4EDByZ4n3ZJbtasmXldu8Xed999JmZ6XH0uX758rp07dyZ4n9XNVNuq+9fvxeOPP27OWWRkpImD1Q26fv36cWK6ePFi936aNm1q3l+pUiVXq1atzHm3urjqTb+D8VnnsF27dq7KlSu7cuXK5Wrbtq05p3o+re7P2mU3vjVr1pjtdRvdVt+jx6xZs6YrJCQkwffn999/N23T7fPkyWPOVfv27d2/S8WLFzefN61dvXrVfH+0DUOHDnWVLl3a/Dx+/Hiv2+/Zs8d9Tnft2pXs496se7me82zZspltVqxYkeD1GTNmuL+P2k1avzP16tVzty0iIiJVvqP79u0z3eP19bJly7oefPBBE2f9t0jbV6VKFfe2es703yrdtkCBAnG+q/rvQlLjERoaar5DV65cSdaQE+3ub7XBk/4O6PP6vfU2HCA+q6t33rx5XbGxse7nZ8+e7f632dPx48fN8/rvkNKY6ONff/01znb6Pn1e9+PNW2+9ZV7v3r37TdsIIG2QdANAOk+6v/32W5Nw6h+0GzZsiPOajtUsUqSI2df69evjvKaPjx49mmB/P/30k/s9W7duveUx3SlNuvWmCd758+e9JgL6eqlSpVx79+6N85p+9uzZs5sLDT///LMrtZJuvWmiePnyZfdrmjgEBwe7k+aePXvG+SN7yZIl7vG4nu/zPJ7eXn311QQxsRJ6jaunF1980TyvCbLGwXLt2jX3BRS9WBEVFRXnfdaxvI37vZUx3cuXL3f98ccfCZ7fvHmz2bcmMb/99lui51CTJc+Y6thXvUDk7TxocpEzZ053ohr/M+mY1Y0bN7ofa8KiFwx0ez0XFy5ccL+mcbEu1OgFEV/Ytm2b+0KM3jShjI6OvmFCp9/j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GTBggK+bAQAAAADwMZJuAAAAAABswkRqqaxz586yYcMGc5s+fbp5bubMmeZ+586d8uKLL8rBgwelatWq5vmyZcua144ePSoDBw6UH374QS5fvizlypWT8ePHS7Nmzdz7Ll68uHTv3l2OHDkin3/+ueTOnVtGjBhhnvMmJiZGunXrJps3b5ZVq1ZJ0aJFvW4XFRVlbpYLFy6Y+9BAlwQFuVLx7CA90fh63mdE169f93UTMsw54lz5N+LsDMTZGYizMxBn/5DU+AW4XK6M+xd3OnT+/Hm57777pGLFijJmzBjz3IEDB0zyXLt2bZk4caLkz59fevbsaZLiTZs2mW327t1rEu769etLaGiozJkzRyZPniyHDx92J8uadF+8eFHGjh0r9957ryxatEiGDx9uknhN3nVM9x133CG7d+82Sfvjjz9unlu5cqU5ZmIiIiJk9OjRCZ5fsGCBhIWF2XauAAAAACCjunLlinTs2NHkgDly5Eh0O5Jum8Z0ayV72rRp7onUGjduLGvWrJGmTZua55YvXy6tW7eWq1evSubMmb3uRxN3Tc779u3rTrobNGggc+fONY81dAULFjQJs25nJd0bN240ibRWr5cuXSo5c+a8YXu9VbrDw8Ol/OCFEh3CkmH+SivcY2vEysgdgRIVmzGXDNsf0cLXTcgQV2BXr14tzZs3l5CQEF83BzYhzs5AnJ2BODsDcfYPmjfly5fvpkk33cvTUOXKld0/FypUyNyfPn3aVLIvXbpkEuVly5bJyZMnJTo62iTkx48fT3QfOlO5Jt26D09a4S5SpIh89913kiVLlpu2SyvreotPE7HoDLp+M5JO45xR1+nmf1K3dq44X/6PODsDcXYG4uwMxDljS2rsmEjNR0HRhFnFxsaa+0GDBsnixYvl1VdfNZVqXfarUqVKcu3atUT3Ye3H2oelVatW8uOPP8qWLVts/DQAAAAAgJuh0m2DTJkymfHat0LHduskbO3btzePtfKt3cWTo1evXqZr+v33328q5w0bNkzWfgAAAAAAKUPSbQMde71161aTNGfLli1BJdqb0qVLy5dffilt27Y11euRI0cm6X2J6devn0n827RpIytWrJC777472fsCAAAAACQP3cttoF3Fg4KCpHz58mbW8Pjjsr15/fXXzRJg9erVM4l3ixYtpHr16ilqx4ABA8wka9rdXJcN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30B3NwMW8fdxyNSmInUi1kpCipdre+yUzm5tFwAAAOBJ6OkGAAAAAMAihG43adOmjYwcOdLdzQAAAAAAWIjQDQAAAACARQjdHiIxMdHdTQAAAAAAZEDodqPU1FR55plnpESJElK2bFmJiIhw7Tt37pwMHjxYSpUqJUFBQdK2bVvZvXu3a78e26BBA3n77belcuXKUrBgQbP9+PHj0q1bNylcuLB53n333Se///67W94fAAAAANgdq5e70YIFC2TUqFESExMjX3/9tQwYMEBatmwp7du3l169ekmhQoVk9erVUrRoUXnjjTckNDRUDh06ZEK6OnLkiPznP/+Rjz/+WHx8fEyIdwbujRs3SnJysjz++OPSu3dv2bBhQ5btSEhIMDenuLg4c+/v7RAfH0cu/CbgDlrftPdOSUlJbmoRrOCsJ3X1bNTZHqizPVBne6DOniG79fNyOBykKjctpJaSkiKbN292bWvatKnp0e7SpYt07txZTp06Jf7+/q791apVMz3jQ4cONT3dL730kvz666+mN1xFRUVJp06d5NixYxISEmK27du3T2699Vb59ttv5bbbbsu0LXquyMjIy7YvXrxYAgICLHj3AAAAAJC/xcfHS9++feX8+fNmlHFW6Ol2o3r16qV7XK5cORO0dRj5xYsXJTg4ON3+S5cuydGjR12PK1as6Arcav/+/SZsOwO3ql27thQrVszsyyp0h4eHmx73tD3deo5JO70l2c8nR94r8h7t4Z7YJFXGbfeWhNT/Xad7b0SYW9uFnP8GVr+Q0xE0fn5+7m4OLEKd7YE62wN1tgfq7BmcI4SvhtDtRhn/wLy8vMwQcQ3cGsAzGxKuAdopMDAwR9qhvelpe9SdNIglp/wvjMEzaZ0T0tSZD37PpHWltp6POtsDdbYH6mwP1Dl/y27tCN15UKNGjeTkyZPi6+srlSpVyvbzatWqJT///LO5pR1erouyaY83AAAAACB3sXp5HtSuXTtp3ry5dO/eXdatWyexsbGydetWGTt2rGzfvv2Kz6tbt6488MAD8t1335l53P369ZPWrVtLkyZNcvU9AAAAAAAI3XmSDjNftWqV3HnnnTJw4ECpUaOG9OnTR3766ScpU6bMFZ+3cuVKKV68uHmuhvAqVarIBx98kKvtBwAAAAD8F8PL3SSz+dorVqxw/VykSBGZPXu2uWW14nja63o7VahQwQTvnBATHnrZYm7wrAU89MsdXTiNuUQAAACANejpBgAAAADAIoRuAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALELoBgAAAADAIoRuAAAAAAAs4mvViZH/NZscLcm+ge5uBizi7+OQqU1F6kSslYQUL7MtdkpndzcLAAAA8Cj0dAMAAAAAYBFCt5sNGDBAunfv7u5mAAAAAAAsQOj2YAR6AAAAAHAv5nR7oJSUFPHy+u8cXQAAAACA+xC6c8myZcskMjJSjhw5IgEBAdKwYUNZuXKla//06dNlxowZkpiYKH369JFZs2aJn5+f2Xf27FkZMWKEfPrpp5KQkCCtW7eW2bNnS/Xq1c3++fPny8iRI+W9996TZ599Vg4dOiQPPvigLFiwwOx3BvD169dLmzZtLmubnlNvTnFxcebe39shPj4Oi38zcBetb9p7lZSU5MYWwQrOmlJbz0ad7YE62wN1tgfq7BmyWz9Cdy44ceKE3H///TJ16lTp0aOHXLhwQTZv3iwOh8MVhsuVK2fuNZT37t1bGjRoIEOGDHENEz98+LB88sknEhQUJGPGjJG7775b9u3b5wrm8fHx8vLLL8vbb78twcHB5nyXLl0yAXrevHnmmBIlSmTavsmTJ5svBDJ6vmGqBASkWPibQV4wsUmq6+dVq1a5tS2wTlRUlLubgFxAne2BOtsDdbYH6py/aQbLDi+HM/nBMt999500btxYYmNjpWLFiun2aaDesGGDHD16VHx8fMy2++67T7y9vWXp0qUmbNeoUUO2bNkiLVq0MPvPnDkjISEhpie7V69epqd74MCBsmvXLqlfv366c587d05WrFhxxfZl1tOt5689eqkk+3HJME+lPdwauMdt95aE1P+OhtgbEebuZsGCb2D1/9Dbt2/v+pIOnoc62wN1tgfqbA/U2TNobipZsqScP3/edI5mhZ7uXKBBODQ0VOrWrSthYWHSoUMH6dmzpxQvXtzsv/XWW12BW2kv9Z49e8zP+/fvF19fX2nWrJlrv/Zk16xZ0+xzKlCggNSrV++62ufv729uGWkQS/7/6zfDc2mdndfp5kPfc2ltqa/no872QJ3tgTrbA3XO37JbO1YvzwUaqPWbrNWrV0vt2rVlzpw5JjQfO3Ys02LpHOzU1P8N+c2OQoUKsXgaAAAAAOQxhO5cooG4ZcuWZu70zp07Tc/08uXLr/q8WrVqSXJyssTExLi26fDygwcPmgB/JfoaupI5AAAAAMA9CN25QAPzSy+9JNu3b5fjx4/Lxx9/LH/88YcJ1FejK5R369bNLKr21Vdfye7du83K5DfddJPZfiWVKlWS77//3gT006dPszoiAAAAAOQyQncu0En1mzZtMiuO66Jozz//vLk8WKdOnbL1fF19XBdi69KlizRv3tyseq6rTF9tDoEGdR3G3qRJEylVqpRZjA0AAAAAkHtYSC0XaI/2mjVrMt2nK49npNfoTksXXNNrcGdFVynXW0YatNetWyfXKyY81CzaBs+kIx/0yxtdsZwFPAAAAABr0NMNAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFuE43stRscrQk+wa6uxmwyOGJHdzdBAAAAMDj0dMNAAAAAIBFCN15QGxsrHh5ecmuXbvc3RQAAAAAQA4idOdTAwYMkO7du7u7GQAAAACAKyB0AwAAAABgEUL3NapUqZLMmjUr3bYGDRpIRESE+VmHic+dO1c6deokhQoVkipVqsiyZcvSHf/tt99Kw4YNpWDBgtKkSRPZuXNnuv0pKSkyaNAgqVy5sjlHzZo15dVXX3Xt19dasGCBrFy50rye3jZs2GD2/fzzz3LfffdJsWLFpESJEtKtWzczfB0AAAAAkPtYvdwC48aNkylTppigvHDhQunTp4/s2bNHatWqJRcvXpQuXbpI+/bt5f3335djx47JiBEj0j0/NTVVbr75Zvnoo48kODhYtm7dKkOHDpVy5cqZQP3000/L/v37JS4uTubNm2eeowE7KSlJwsLCpHnz5rJ582bx9fWVSZMmSceOHeX777+XAgUKZNrehIQEc3PS8yp/b4f4+Dgs/V3BffTfS9p7eCbqbA/U2R6osz1QZ3ugzp4hu/UjdFugV69eMnjwYPPzxIkTJSoqSubMmSOvv/66LF682ITqd955x/R033rrrfLLL7/Io48+6nq+n5+fREZGuh5rj/fXX38tH374oQndhQsXNj3gGpTLli3rOk5DvJ777bffNr3fSkO59nprT3iHDplfImry5MnpXs/p+YapEhCQkqO/G+Qd+u8y7T08G3W2B+psD9TZHqizPVDn/C0+Pj5bxxG6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BE3FihUdQ4YMuexLUuqc92VWY73Nmzfvmj6rY2NjHZ06dXIUKlTIfLmqX7omJSW54R3heup8/PhxE7BLlChhPrerVavmGD16tOP8+fPpzkOd87aHH37YfB7rf3fp57P+/68zcCv+lu3LS//H3b3tAAAAAAB4IuZ0AwAAAABgEUI3AAAAAAAWIXQDAAAAAGARQjcAAAAAABYhdAMAAAAAYBFCNwAAAAAAFiF0AwAAAABgEUI3AAAAAAAWIXQDAAAAAGARQjcAALhm8+fPFy8vr0xvzz77rLubBwBAnuHr7gYAAID8a8KECVK5cuV02+rUqeO29gAAkNcQugEAwHXr1KmTNGnSRPKb+Ph4CQgIcHczAAA2wPByAACQqw4fPiz/+Mc/pGzZslKwYEG5+eabpU+fPnL+/Pl0x73//vvStGlTE46LFy8ud955p6xbty7dMa+//rrceuut4u/vL+XLl5fHH39czp07l+6YNm3amN73HTt2mHPo+Z577jmzLyEhQV544QWpVq2aOUdISIg888wzZjsAADmBnm4AAHDdNCifPn063baSJUtmeXxiYqKEhYWZUPvPf/7TBO9ff/1VPvvsMxOWixYtao6LjIyUiIgIadGihRnCXqBAAYmJiZEvv/xSOnToYI7R/Xpcu3bt5NFHH5WDBw/K3LlzZdu2bbJlyxbx8/Nzve6ZM2dMr7yG+wcffFDKlCkjqampcs8998hXX30lQ4cOlVq1asmePXtk5syZcujQIVmxYoVlvzcAgH0QugEAwHXTwJuRw+HI8vh9+/bJsWPH5KOPPpKePXu6to8fP97185EjR0zQ7tGjhyxbtky8vb0vO/cff/whkydPNgF89erVrmNuueUWGT58uOklHzhwoOt5J0+elH//+9/yyCOPuLbpMV988YVs3LhR7rjjDtd27RUfNmyYbN261YR+AABuBMPLAQDAdfvXv/4lUVFR6W5X4uzJXrt2rZlXnRntYdZeaA3iaQO30tXRlYZl7TUfOXJkumOGDBkiQUFB8vnnn6d7ng4dTxvClQZ/7d3WoK699c5b27Ztzf7169df0+8CAIDM0NMNAACum865vpaF1HSl81GjRskrr7wiixYtklatWpkh3jrk2xnIjx49aoJ07dq1szzPTz/9ZO5r1qyZbrsOQ69SpYprv9NNN91k9mWcW75//34pVapUpq9x6tSpbL8vAACyQugGAAC5asaMGTJgwABZuXKlWRjtiSeeMEPFv/nmG7OomhUKFSp02TbtTa9bt675AiAzuqgaAAA3itANAABynYZdvT3//PNm7nTLli3NnOtJkyZJ1apVTSDW+d8NGjTI9PkVK1Y097p4mvZsO+mQc50zntlc84z0dXbv3i2hoaGuYesAAOQ05nQDAIBcExcXJ8nJyem2afjW4eTOy3R1797dPNbF1DR8p+VcSE1DtQ4Xnz17drqF29555x2zonrnzp2v2pb77rvPrJz+1ltvXbbv0qVL8tdff133+wQAwImebgAAkGv0kl+6univXr2kRo0aJoAvXLhQfHx8zLW7lV4ze+zYsTJx4kQz5/vee+81C6HppcD0Wtw6FF3nYYeHh5tLhnXs2NHMC9deb71u92233WbmiF/NQw89JB9++KFZqVwXTdPe9pSUFDlw4IDZrou9Xct8dQAAMkPoBgAAuaZ+/frmOt2ffvqp6WUOCAgw2/SyX7fffrvrOO3l1kXX5syZYwK4HlevXj0TlJ30Ot0avl977TV58sknpUSJEuZ62y+99FK6a3RnRXvTdaV0vS73e++9J8uXLzevo8PVR4wYYb4UAADgRnk5rnQxTQAAAAAAcN2Y0w0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAIg1/g9xK0fTLxTQmQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x1200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot feature importance\n",
    "fig, ax = plt.subplots(figsize=(10, 12))\n",
    "xgb.plot_importance(\n",
    "    gs_bow_pipe.best_estimator_.named_steps['xgbregressor'],\n",
    "    max_num_features=50,\n",
    "    ax=ax,\n",
    "    importance_type='gain',\n",
    "    show_values=False,\n",
    "    height=0.6  )\n",
    "\n",
    "tick_labels = [item.get_text() for item in ax.get_yticklabels()]\n",
    "\n",
    "passthrough_columns = X_train.drop(columns=['tokenized_words']).columns.tolist()\n",
    "\n",
    "for i in range(len(tick_labels)):\n",
    "    try:\n",
    "        feature_index = int(tick_labels[i][1:])\n",
    "\n",
    "        if feature_index < num_bow_features:\n",
    "            tick_labels[i] = feature_names[feature_index]\n",
    "        else:\n",
    "            passthrough_index = feature_index - num_bow_features\n",
    "\n",
    "            if 0 <= passthrough_index < len(passthrough_columns):\n",
    "                tick_labels[i] = passthrough_columns[passthrough_index]\n",
    "\n",
    "    except ValueError:\n",
    "        pass\n",
    "\n",
    "# Set updated tick labels\n",
    "ax.set_yticklabels(tick_labels)\n",
    "\n",
    "plt.title(\"Feature Importance - XGBoost (BOW)\", fontsize=16)\n",
    "plt.xlabel(\"F score\", fontsize=12)\n",
    "plt.ylabel(\"Features\", fontsize=12)\n",
    "plt.xticks(fontsize=10)\n",
    "plt.yticks(fontsize=10)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"rf_bow.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 47917,
     "status": "ok",
     "timestamp": 1746042288901,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "AXUD15CUJODL",
    "outputId": "7dfba095-60d6-4c4a-d8f6-e0b9d81d87d1"
   },
   "outputs": [],
   "source": [
    "# predict\n",
    "predictions = gs_bow_pipe.predict(X_test)\n",
    "predictions = list(map(round,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[157888  60574]\n",
      " [ 10448  27608]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.72      0.82    218462\n",
      "           1       0.31      0.73      0.44     38056\n",
      "\n",
      "    accuracy                           0.72    256518\n",
      "   macro avg       0.63      0.72      0.63    256518\n",
      "weighted avg       0.85      0.72      0.76    256518\n",
      "\n",
      "Specificity : 0.7227252336790838\n",
      "ROC-AUC : 0.7240912273083246\n"
     ]
    }
   ],
   "source": [
    "# evaluate\n",
    "cm = confusion_matrix(y_test, predictions)\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()\n",
    "print(confusion_matrix(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Specificity :\", tn/(fp+tn))\n",
    "print(\"ROC-AUC :\", roc_auc_score(y_test, predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uDc3WscPJzfz"
   },
   "source": [
    "### Word2Vec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "background_save": true
    },
    "id": "205lKvKpgDS0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NumPy: 1.26.4\n",
      "Gensim: 4.3.3\n"
     ]
    }
   ],
   "source": [
    "print(f\"NumPy: {numpy.__version__}\\nGensim: {gensim.__version__}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "baEE5GunJ-DD"
   },
   "outputs": [],
   "source": [
    "# Custom transformer for Word2Vec averaging\n",
    "class Word2VecVectorizer(BaseEstimator, TransformerMixin):\n",
    "    def __init__(self, vector_size=100, window=5, min_count=1, workers=1, seed=229):\n",
    "        self.vector_size = vector_size\n",
    "        self.window = window\n",
    "        self.min_count = min_count\n",
    "        self.workers = workers\n",
    "        self.seed = seed\n",
    "        self.model = None\n",
    "\n",
    "    def fit(self, X, y=None):\n",
    "        # Handle different input types\n",
    "        if isinstance(X, np.ndarray):\n",
    "            sentences = [doc.split() for doc in X.ravel()]\n",
    "        elif isinstance(X, pd.DataFrame):\n",
    "            sentences = [doc.split() for doc in X.iloc[:, 0]]\n",
    "        else:\n",
    "            sentences = [doc.split() for doc in X]\n",
    "\n",
    "        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,\n",
    "                              min_count=self.min_count, workers=self.workers, seed=self.seed)\n",
    "        return self\n",
    "\n",
    "    def transform(self, X):\n",
    "        # Handle different input types\n",
    "        if isinstance(X, np.ndarray):\n",
    "            documents = pd.Series(X.ravel())\n",
    "        elif isinstance(X, pd.DataFrame):\n",
    "            documents = X.iloc[:, 0]\n",
    "        else:\n",
    "            documents = X\n",
    "\n",
    "        def document_vector(doc):\n",
    "            words = [word for word in doc.split() if word in self.model.wv]\n",
    "            if not words:\n",
    "                return np.zeros(self.vector_size)\n",
    "            return np.mean(self.model.wv[words], axis=0)\n",
    "\n",
    "        return np.vstack(documents.apply(document_vector))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "i6HfHj8edYkN"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 1 folds for each of 8 candidates, totalling 8 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x110c19d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10741dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x102e1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=nan total time=   2.9s\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10a11dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=nan total time=   3.0s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=300;, score=nan total time=   3.0s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.244 total time= 7.6min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107bcdd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=300;, score=nan total time=   3.0s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=nan total time=   2.8s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.246 total time= 7.6min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10661dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=300;, score=nan total time=   2.4s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=6, xgbregressor__n_estimators=300;, score=0.239 total time= 7.8min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106e19d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: #000;\n",
       "  --sklearn-color-text-muted: #666;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: flex;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "  align-items: start;\n",
       "  justify-content: space-between;\n",
       "  gap: 0.5em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label .caption {\n",
       "  font-size: 0.6rem;\n",
       "  font-weight: lighter;\n",
       "  color: var(--sklearn-color-text-muted);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 0.5em;\n",
       "  text-align: center;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.13, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;word2vec&#x27;,\n",
       "                                                                         Word2VecVectorizer(min_count=2,\n",
       "                                                                                            workers=4),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 300)},\n",
       "             verbose=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>GridSearchCV</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html\">?<span>Documentation for GridSearchCV</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.13, train_size=None),\n",
       "             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                                          transformers=[(&#x27;word2vec&#x27;,\n",
       "                                                                         Word2VecVectorizer(min_count=2,\n",
       "                                                                                            workers=4),\n",
       "                                                                         &#x27;tokenized_words&#x27;),\n",
       "                                                                        (&#x27;standardscaler&#x27;,\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         [&#x27;user_reviews&#x27;,\n",
       "                                                                          &#x27;days_since_review&#x27;,\n",
       "                                                                          &#x27;user_rating&#x27;,...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={&#x27;xgbregressor__learning_rate&#x27;: (0.1, 0.3),\n",
       "                         &#x27;xgbregressor__max_depth&#x27;: (4, 6),\n",
       "                         &#x27;xgbregressor__n_estimators&#x27;: (100, 300)},\n",
       "             verbose=3)</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>best_estimator_: Pipeline</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,\n",
       "                 ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                                   transformers=[(&#x27;word2vec&#x27;,\n",
       "                                                  Word2VecVectorizer(min_count=2,\n",
       "                                                                     workers=4),\n",
       "                                                  &#x27;tokenized_words&#x27;),\n",
       "                                                 (&#x27;standardscaler&#x27;,\n",
       "                                                  StandardScaler(),\n",
       "                                                  [&#x27;user_reviews&#x27;,\n",
       "                                                   &#x27;days_since_review&#x27;,\n",
       "                                                   &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;,\n",
       "                                                   &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;,\n",
       "                                                   &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                                   &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;,\n",
       "                                                   &#x27;quote&#x27;, &#x27;...\n",
       "                              feature_types=None, feature_weights=None,\n",
       "                              gamma=None, grow_policy=None,\n",
       "                              importance_type=None,\n",
       "                              interaction_constraints=None, learning_rate=0.1,\n",
       "                              max_bin=None, max_cat_threshold=None,\n",
       "                              max_cat_to_onehot=None, max_delta_step=None,\n",
       "                              max_depth=6, max_leaves=None,\n",
       "                              min_child_weight=None, missing=nan,\n",
       "                              monotone_constraints=None, multi_strategy=None,\n",
       "                              n_estimators=300, n_jobs=-1,\n",
       "                              num_parallel_tree=None, ...))])</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>columntransformer: ColumnTransformer</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html\">?<span>Documentation for columntransformer: ColumnTransformer</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;,\n",
       "                  transformers=[(&#x27;word2vec&#x27;,\n",
       "                                 Word2VecVectorizer(min_count=2, workers=4),\n",
       "                                 &#x27;tokenized_words&#x27;),\n",
       "                                (&#x27;standardscaler&#x27;, StandardScaler(),\n",
       "                                 [&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;,\n",
       "                                  &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;,\n",
       "                                  &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;,\n",
       "                                  &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;,\n",
       "                                  &#x27;sentiment&#x27;])])</pre></div> </div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>word2vec</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>tokenized_words</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>Word2VecVectorizer</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>Word2VecVectorizer(min_count=2, workers=4)</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>standardscaler</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[&#x27;user_reviews&#x27;, &#x27;days_since_review&#x27;, &#x27;user_rating&#x27;, &#x27;rating_diff&#x27;, &#x27;num_words&#x27;, &#x27;avg_word_len&#x27;, &#x27;avg_sent_len&#x27;, &#x27;pct_verbs&#x27;, &#x27;pct_nouns&#x27;, &#x27;pct_adj&#x27;, &#x27;quote&#x27;, &#x27;sentiment&#x27;]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>StandardScaler</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html\">?<span>Documentation for StandardScaler</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>StandardScaler()</pre></div> </div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>remainder</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>[]</pre></div> </div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>passthrough</div></div></label><div class=\"sk-toggleable__content fitted\"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>StandardScaler</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.StandardScaler.html\">?<span>Documentation for StandardScaler</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>StandardScaler()</pre></div> </div></div><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" ><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>XGBRegressor</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.XGBRegressor\">?<span>Documentation for XGBRegressor</span></a></div></label><div class=\"sk-toggleable__content fitted\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=&#x27;error&#x27;, feature_types=None,\n",
       "             feature_weights=None, gamma=None, grow_policy=None,\n",
       "             importance_type=None, interaction_constraints=None,\n",
       "             learning_rate=0.1, max_bin=None, max_cat_threshold=None,\n",
       "             max_cat_to_onehot=None, max_delta_step=None, max_depth=6,\n",
       "             max_leaves=None, min_child_weight=None, missing=nan,\n",
       "             monotone_constraints=None, multi_strategy=None, n_estimators=300,\n",
       "             n_jobs=-1, num_parallel_tree=None, ...)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=ShuffleSplit(n_splits=1, random_state=229, test_size=0.13, train_size=None),\n",
       "             estimator=Pipeline(steps=[('columntransformer',\n",
       "                                        ColumnTransformer(remainder='passthrough',\n",
       "                                                          transformers=[('word2vec',\n",
       "                                                                         Word2VecVectorizer(min_count=2,\n",
       "                                                                                            workers=4),\n",
       "                                                                         'tokenized_words'),\n",
       "                                                                        ('standardscaler',\n",
       "                                                                         StandardScaler(),\n",
       "                                                                         ['user_reviews',\n",
       "                                                                          'days_since_review',\n",
       "                                                                          'user_rating',...\n",
       "                                                     max_cat_to_onehot=None,\n",
       "                                                     max_delta_step=None,\n",
       "                                                     max_depth=None,\n",
       "                                                     max_leaves=None,\n",
       "                                                     min_child_weight=None,\n",
       "                                                     missing=nan,\n",
       "                                                     monotone_constraints=None,\n",
       "                                                     multi_strategy=None,\n",
       "                                                     n_estimators=None,\n",
       "                                                     n_jobs=-1,\n",
       "                                                     num_parallel_tree=None, ...))]),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'xgbregressor__learning_rate': (0.1, 0.3),\n",
       "                         'xgbregressor__max_depth': (4, 6),\n",
       "                         'xgbregressor__n_estimators': (100, 300)},\n",
       "             verbose=3)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=300;, score=nan total time=   3.0s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=0.237 total time= 7.5min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10381dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=100;, score=0.247 total time= 7.6min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x113a1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=4, xgbregressor__n_estimators=300;, score=0.248 total time= 7.7min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104bb5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=100;, score=nan total time=   3.1s\n",
      "[CV 1/1] END xgbregressor__learning_rate=0.3, xgbregressor__max_depth=4, xgbregressor__n_estimators=300;, score=0.250 total time= 7.9min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104b1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[CV 1/1] END xgbregressor__learning_rate=0.1, xgbregressor__max_depth=6, xgbregressor__n_estimators=300;, score=0.254 total time= 8.0min\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x11121dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    }
   ],
   "source": [
    "# Pipeline\n",
    "w2v_pipe = make_pipeline(\n",
    "    ColumnTransformer( transformers=[\n",
    "            ('word2vec', Word2VecVectorizer(vector_size=100, window=5, min_count=2, workers=4, seed=229), 'tokenized_words'),\n",
    "            ('standardscaler', StandardScaler(), numerical_cols) \n",
    "        ], remainder = 'passthrough'\n",
    "    ),\n",
    "    StandardScaler(),\n",
    "    xgb.XGBRegressor(\n",
    "        objective='binary:logistic',\n",
    "        eval_metric='error',\n",
    "        seed=229,\n",
    "        n_jobs=-1\n",
    "    )\n",
    ")\n",
    "\n",
    "# Hyperparameter grid\n",
    "parameters = {\n",
    "    'xgbregressor__n_estimators': (100, 300),\n",
    "    'xgbregressor__max_depth': (4, 6),\n",
    "    'xgbregressor__learning_rate': (0.1, 0.3)\n",
    "}\n",
    "\n",
    "# Grid search with ShuffleSplit\n",
    "gs_w2v_pipe = GridSearchCV(\n",
    "    w2v_pipe,\n",
    "    parameters,\n",
    "    cv=ShuffleSplit(n_splits=1, test_size=0.13, random_state=229),n_jobs=-1,\n",
    "    verbose=3\n",
    ")\n",
    "gs_w2v_pipe.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "wVLh7UjXnY6z"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([441.87663484, 455.40721107, 447.57951093, 471.7952199 ,\n",
      "       448.33307099, 467.57619596, 449.62286305, 463.6331563 ]), 'std_fit_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'mean_score_time': array([9.59704709, 7.55060983, 8.60207415, 7.04219079, 7.92727518,\n",
      "       6.97115397, 7.50284195, 6.41956973]), 'std_score_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'param_xgbregressor__learning_rate': masked_array(data=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=1e+20), 'param_xgbregressor__max_depth': masked_array(data=[4, 4, 6, 6, 4, 4, 6, 6],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'param_xgbregressor__n_estimators': masked_array(data=[100, 300, 100, 300, 100, 300, 100, 300],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'params': [{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 300}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 300}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 300}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 300}], 'split0_test_score': array([0.23730022, 0.24766922, 0.24684417, 0.2538259 , 0.24357611,\n",
      "       0.24956292, 0.24596804, 0.23899508]), 'mean_test_score': array([0.23730022, 0.24766922, 0.24684417, 0.2538259 , 0.24357611,\n",
      "       0.24956292, 0.24596804, 0.23899508]), 'std_test_score': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'rank_test_score': array([8, 3, 4, 1, 6, 2, 5, 7], dtype=int32)}\n",
      "{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 300}\n",
      "\n",
      "Best model saved as 'xgboost_w2v_model_cde.pkl'\n"
     ]
    }
   ],
   "source": [
    "# print model\n",
    "\n",
    "print(gs_w2v_pipe.cv_results_)\n",
    "print(gs_w2v_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./xgboost_w2v_model_cde.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_w2v_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'xgboost_w2v_model_cde.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "qcKB9M2Kf24T"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of features (Word2Vec vector size): 112\n"
     ]
    }
   ],
   "source": [
    "# Feature Importance\n",
    "sorted_ind = gs_w2v_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_.argsort()[::-1]\n",
    "num_features = len(sorted_ind)\n",
    "print(f\"Number of features (Word2Vec vector size): {num_features}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "Bk_7smtVf-G-"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Top 10 embedding dimensions: [100 104  98 102 101 110  27   0  58  94]\n",
      "Top 10 importances: [0.14409606 0.09460725 0.03902172 0.02292424 0.01697556 0.01688075\n",
      " 0.01594122 0.01586273 0.01484742 0.01436567]\n"
     ]
    }
   ],
   "source": [
    "# The top 10 most important embeddings\n",
    "top_10_indices = sorted_ind[:10]\n",
    "print(\"Top 10 embedding dimensions:\", top_10_indices)\n",
    "print(\"Top 10 importances:\", gs_w2v_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_[top_10_indices])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "PSNzCKMBdj3S"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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AT8f9JkaA36nekFI8I7Rj7vvvjtkahT3T11GKHHx7bff2p8xV+PAz33dunVD/r4NrpegRqJ9+/b2dwzvjLfv9IlzsDPt8i7wvvGZSOB9uuKKK+zXBw4cCDhGHy6WGvWvf/3L/P777/Zn/L777guwzdSpU639uHf6yjPv0qVLnOsk5ffAE//fNm+88YbJkKR26EQIkTnSol599dXYrFmz2s9UqFAh9uGHH46tVatWbM6cOe0+UieCad++vT1WunTp2IYNG9o0jOrVq/vaefrppwPOnzJlSuxDDz1kj+XNm9em0ni3hKbYuHQN72e8+xs3bhx7ww03xBYsWDD2H//4R2yzZs1iW7Vq5TuvT58+9jz6WbVqVXu8WrVqNr2DdIV33nknNhppUa4/hPVvueWW2Kuuusref4MGDWw6BMe6d+8eu2fPntgiRYrEFitWzKYa1K5d2/aD46NHjw57vZ49e9p7KFOmTGyLFi1sCoWzfe/eveN8jhB/vXr17HHSOe699177zLgu+6644orYLVu2xPkcx1xfaZ/34tFHH7U2279/v30OLn2nZs2aAc/0X//6l6+dunXr2s+XK1fOpoxg90qVKvna5x0MxtmwadOmNo3ksssui23SpIm1KfZ0aTt//PFHnM+uXLnSns85nMtnuObtt98emyNHjjjvz08//WT7xvmFChWytvrnP//p+1kqXry4vd+UhjQN3h/68Nxzz9lUE74eNWpUyPO3bdvms+mXX34Z8XUvlhaFzfPly2fPWbZsWZzjb7/9tu99rFixon1natSo4evb0KFDo/KOfvXVVzati+M333xz7IMPPmifM7+L6F/58uV952IzfldxbuHChQPeVX4vJPR5kNbDO3T69OmQ5/AzHOrnd9iwYXb/kiVLbNpb9uzZ7bleNmzY4Lt/b1rO0qVL7T53n/zMuxQr9rVr1y5sSlPLli3tO83vGX4HYSN3v6dOnYq94447fL+X77//fms/7HP55ZfHtm7dOtFpn7/++quvX6ShBaficd/xvZ/169e3x5999lnfvuPHj/vSrXiu/B7k7xS2YB99DdVeYn8POEiN4jOHDh2KzWhIXAghkl1cfPLJJ9ax5g/3mjVrAo6RS33ttdfatlavXh1wjO/37t0bp71vv/3W95kvvvgi0TUXSRUXbDiyf/75Z0iHh+M33nhj7Pbt2wOOce/5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rGFF1/z5803p0qVN7ty5Tbly5czHH38ccDw2NtY8//zz5uqrrzaXXHKJqVevnvn++++Tte9CCCGEEKlNmhIXgJDAwfNu7733nsko4GheddVVKXKtAgUKWPsdPHjQfPbZZ6Zjx45m1qxZNopz6NAh33mFChUy+fPnN2mdIkWKmFy5ciXrNc6ePWuuvPJKM2jQIFO+fPmQ52DLRx991LRv395s3brVNG3a1G47d+70nTN27FgzYcIE89Zbb5kvvvjC5M2b1zRs2NCcOXMmWfsvhBBCCJGapDlxgfOIE+ndGHV3MBI/efJkc//995s8efKYW265xXz++edmz549Nr0HJ65GjRpm7969cdrmc8WKFbOfa968eZwR/KlTp9r2GI1mVHrSpEkBxzdu3GgqVqxoj1epUsU6lsEwgs2oNyKidu3aNnXKS3BalEvZmj17tilevLi59NJLTYsWLcyJEyd85/B1q1at7L0xEv7KK68kKJUJW2E/PsN94QzjGJ88edL069fPd15wW/RjxIgRpnXr1iZfvnzm+uuvN4sXLza//vqreeCBB+y+2267zWzevDngeuvWrTN33nmnvXfs3LNnT3Pq1KmAdkeOHGnatWtnxcx1111n3n77bd/xc+fOme7du9v+YmOuO2rUqLBpUV999ZWpU6eOvd7ll19uxRP35o2E4fSPGzfOtsk53bp1M3/99VdYm9HH1157zd47zyIUHEcE9+3b19qVqE+lSpXM66+/7otavPrqq1agYC9shahD0CmtSwghhBAZmXQ5WxTO3Pjx4+327LPP2jSWEiVKmP79+1uHFecVJ3XZsmW+zyA+PvjgA7NkyRJz/Phx62h37drVzJkzxx7nf9JYcBAREAiHDh06WIe+TZs21mlF0NSvX9+8++675ocffjBPPfVUQL8OHDhgHnzwQevA4ujifPfp0+ei94MQwulcunSp+f33363wGT16tHnxxRft8d69e5v169dbB79w4cK2n19++WVEdSRETRAq77zzjjl//rxNmQoFAgYhMHjwYPv1448/bkUbtn3ppZes3XHAv/76a+v0cw843IgS2kaI8AzYpk+f7mv35Zdfts9vwIAB5sMPPzRdunQxd999t7n55pvtSD/3yHPiOWJPtlAgWogEVK9e3WzatMkcOXLEPPnkk/Z6CDjHp59+aoUF//MOkJaG3Xi2kYKY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Setgx0WLFqV2N4QQQgghRHoQF0888YR1IIO3Ro0amYzCI488Ynbv3p3s10HEOPtly5bNFCxY0FSrVs0MGzbM/PnnnwHnLly40AwfPtykdX7++Wdz7733Jvs1WrZsaW666SaTNWvWsKJr/vz5pnTp0iZ37tymXLly5uOPPw7bZufOne1zePXVV5Ox50IIIYQQqU+aEheAkMDB827vvfeeyShccskl5qqrrkqRaxUoUMDa7+DBg+azzz4zHTt2NLNmzbJRnEOHDvnOK1SokMmfP79J6xQpUsTkypUrWa9x9uxZc+WVV5pBgwaZ8uXLhzwHWz766KOmffv2ZuvWraZp06Z227lzZ5xz//Wvf5kNGzaYokWLJmu/hRBCCCHSAmlOXOA84kR6N0bdHYwAT5482dx///0mT5485pZbbjGff/652bNnj03vyZs3r6lRo4bZu3dvnLb5XLFixeznmjdvHmcEf+rUqbY9RqMZlZ40aVLA8Y0bN5qKFSva41WqVLGOZTCMYDPqjYioXbu2TZ3yEpwW5VK2Zs+ebYoXL24uvfRS06JFC3PixAnfOXzdqlUre29XX321eeWVVxKUyoStsB+f4b5whnGMT548afr16+c7L7gt+jFixAjTunVrky9fPnP99debxYsXm19//dU88MADdt9tt91mNm/eHHC9devWmTvvvNPeO3bu2bOnOXXqVEC7I0eONO3atbNi5rrrrjNvv/227/i5c+dM9+7dbX+xMdcdNWpU2LSor776ytSpU8de7/LLL7fiiXvzRsJw+seNG2fb5Jxu3bqZv/76K6zN6ONrr71m751nEQqOI4L79u1r7UrUp1KlSub1118POO+nn34yPXr0MHPmzDE5cuSI91kJIYQQQmQE0py4SAg4czh/1DogAkhj6dSpk+nfv791eGNjY62T6gXx8cEHH5glS5aYTz75xAqDrl27+o7jAD7//PPmxRdfNN988411ggcPHmxmzpxpj+O0ImjKlCljtmzZYkXBM888E3CNAwcOmAcffNA0adLE9u3JJ580zz333EXvByGE07x06VK7rVmzxowePdp3vHfv3mb9+vXWwV+xYoVZu3at+fLLLyOyHVEThAptnT9/Pux5CJiaNWtaOzVu3Ng8/vjj1uaPPfaYvXbJkiXt99ja3QMO90MPPWR27Nhh5s2bZ8VG8HN4+eWXfcIM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u/v37zdZsmQx27ZtC3vO6tWr7Tl//PGH/X7GjBnmsssuM2mJe+65x/Tq1cukdbDjokWLUrsbQgghhBCZlnQlLp544gnrQAZvjRo1MhmFRx55xOzevTvZr4OIcfbLli2bKViwoKlWrZoZNmyY+fPPPwPOXbhwoRk+fLhJ6/z888/m3nvvTfbrvPHGG+aWW24xl1xyibn55pvNrFmz4pzz6quv2mOcU6xYMfP000+bM2fOJHvfhBBCCCFSk3RXc4GQmD59esC+XLlymYwCzihbSlCgQAHz3XffmdjYWBs5+eyzz8yoUaOsfdevX2+KFi1qzytUqJBJDxQpUiTZr/Hmm2+a/v37mylTppjbb7/dbNy40XTo0MGKsyZNmthz5s6da5577jnzzjvvmBo1alix6ITx+PHjk72PQgghhBCpRbqKXDghgRPp3XDsHDhwkydPNvfff7/JkyePHWH+/PPPzZ49e2x6T968ea3Dt3fv3jht8zlGmflc8+bN44zgT5061baXO3duU7p0aTNp0qSA4ziaFStWtMerVKlitm7dGucaH3/8sbnpppusgKhdu7ZNnfISnBblUrZmz55tihcvbi699FLTokULc+LECd85fN2qVSt7b1dffbV55ZVXEpTKhK2wH5/hvtq3b28FxsmTJ02/fv185wW3RT9GjBhhWrdubfLly2euv/56s3jxYvPrr7+aBx54wO677bbbzObNmwOut27dOnPnnXf6RvN79uxpTp06FdDuyJEjTbt27Uz+/PnNddddZ95++23f8XPnzpnu3bvb/mJjrosYCpcW9dVXX5k6derY611++eWmY8eO9t4cOPxNmzY148aNs21yTrdu3cxff/0V1mY8h06dOtkIU4kSJeyzoN0xY8b4zsGGNWvWNC1btrT31KBBA/Poo4/a90MIIYQQIiOT7sRFQiCFB8eXWgdEAE4eDiEjzji8jNTjpHpBfHzwwQdmyZIl5pNPPrHCoGvXrr7jc+bMMc8//7x58cUXzTfffGOd4MGDB5uZM2fa4zitCJoyZcqYLVu2WFHwzDPPBFzjwIED5sEHH7Qj3PTtySeftCPcFwMhhNO8dOlSu61Zs8aMHj3ad7x379420oCDv2LFCrN27Vrz5ZdfRmS7q666ygoV2jp//nzY8xAwONDYqXHjxubxxx+3Nn/sscfstUuWLGm/x9buHog6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csUVV6R2t4QQQgghhPAhcZEOoAaA3H8hhBBCCCHSMmlutighhBBCCCFE+kTiQgghhBBCCBEVJC6EEEIIIYQQUUHiQgghhBBCCBEVJC6EEEIIIYQQUUHiQgghhBBCCBEVJC6EEEIIIYQQUUHrXAiRhqk2apWJyZ7XZHT2j26c2l0QQgghRBTIFJGLoUOHmgoVKkS93f3795ssWbKYbdu2hT1n9erV9pw//vjDfj9jxgxz2WWXmbTEPffcY3r16mXSOthx0aJFqd0NIYQQQgiRHsTFE088YR3I4K1Ro0Ymo/DII4+Y3bt3J/t1EDHOftmyZTMFCxY01apVM8OGDTN//vlnwLkLFy40w4cPN2mdn3/+2dx7773Jfo2WLVuam266yWTNmjWs6Jo/f74pXbq0yZ07tylXrpz5+OOP49i0QYMG5vLLL7+oABVCCCGEyCikKXEBCAkcPO/23nvvmYzCJZdcYq666qoUuVaBAgWs/Q4ePGg+++wz07FjRzNr1iwbxTl06JDvvEKFCpn8+fObtE6RIkVMrly5kvUaZ8+eNVdeeaUZNGiQKV++fMhzsOWjjz5q2rdvb7Zu3WqaNm1qt507d/rOOXXqlKlVq5YZM2ZMsvZXCCGEECItkebEBc4jTqR3Y9TdwSjw5MmTzf3332/y5MljbrnlFvP555+bPXv22PSevHnzmho1api9e/fGaZvPFStWzH6uefPmcUbwp06dattjNJpR6UmTJgUc37hxo6lYsaI9XqVKFetYBsMINqPeiIjatWvb1CkvwWlRLmVr9uzZpnjx4ubSSy81LVq0MCdOnPCdw9etWrWy93b11VebV155JUGpTNgK+/EZ7gtnGMf45MmTpl+/fr7zgtuiHyNGjDCtW7c2+fLlM9dff71ZvHix+fXXX80DDzxg9912221m8+bNAddbt26dufPOO+29Y+eePXtaJ9vb7siRI027du2smLnuuuvM22+/7Tt+7tw50717d9tfbMx1R40aFTYt6quvvjJ16tSx1yNCgHji3ryRMJz+cePG2TY5p1u3buavv/4KazP6+Nprr9l751mEguOI4L59+1q7EvWpVKmSef31133nPP744+b555839erVi/cZCSGEEEJkJNKcuEgIOHM4f6SaIAJIY+nUqZPp37+/dXhjY2Otk+oF8fHBBx+YJUuWmE8++cQKg65du/qO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Rq0yMdnzmvTM/tGNU7sLQgghhEjLNRcZjaFDh5oKFSpEvd39+/fb9LBt27bFK9Q4548//rDfz5gxw1x22WUmLXHPPfeYXr16mbQOdly0aFFqd0MIIYQQItOSIHHxv//7v3YL/v5iW7R54oknrAMZvDVq1MhkFB555BGze/fuZL8OIsbZj3qZggULmmrVqplhw4aZP//8M+DchQsXmuHDh5u0zs8//2zuvffeZL/OG2+8YVemv+SSS+x6L7NmzQp7LhMfYOOmTZsme7+EEEIIIdJFWhQj1zhI//d//2dy5szp+z4csbGx9jgreEcbhMT06dMD9uXKlctkFHBY2VKCAgUKmO+++84+LyInn332mRk1apS17/r1603RokV9NTbpgSJFiiT7Nd58803Tv39/M2XKFDsl88aNG+2kBoizJk2axIlcPfPMM+bOO+9M9n4JIYQQQqSbyMWnn35q/vOf/1hh4f0+3OaOJwcICZxI74Zj50DUTJ482dx///0mT548doT5888/N3v27LGiKG/evKZGjRpm7969cdrmc8WKFbOfa968eZwR/KlTp9r2cufObUqXLm0mTZoUcBxHs2LFivZ4lSpVzNatW+Nc4+OPP7aLDyIgateubR1QL8FpUS5la/bs2aZ48eLm0ksvNS1atDAnTpzwncPXzODFvV199dXmlVdeSVAqE7bCfnyG+2rfvr0VGCdPnjT9+vXznRfcFv0YMWKEad26tcmXL5+5/vrrzeLFi82vv/5qHnjgAbvvtttuM5s3bw643rp166yjzb1j5549e5pTp04FtDty5EjTrl07kz9/fnPdddeZt99+23f83Llzpnv37ra/2JjrIobCpUV99dVXpk6dOvZ6TEDQsWNHe2/eSBgRhXHjxtk2Oadbt27mr7/+CmsznkOnTp1shKlEiRL2WdDumDFjAs5DWPNMXnjhBXueEEIIIURmIEHi4u6777Zb8PcX21ILUnhwfKl1QAS0bNnSOoSMOOPwMlKPk+oF8cG6HUuWLDGffPKJFQZdu3b1HZ8zZ455/vnnzYsvvmi++eYb6wQPHjzYzJw50x7HaUXQlClTxmzZssWKAkatvRw4cMA8+OCDdoSbvj355JPmueeeu+j9IIRwmpcuXWq3NWvWmNGjR/uO9+7d20YacPBXrFhh1q5da7788suIbHfVVVdZp5i24os8IWBq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69atzSuvvGKaNGlifvrpJ9O5c2fToUMHK8JcNOrRRx81NWrUMLlz5zZjxowxDRo0MF9//bW55pprkr2PQgghhBBpOnJx9913x9ny5s1r034Yva1YsaLd+Jp9+fLlM3fddVeydBghgRPp3Rgd9oqayZMnm/vvv9/255ZbbjGff/652bNnj03vod84faGED58rVqyY/Vzz5s3jjOBPnTrVtofDWLp0aTNp0qSA44xOYweOV6lSxa5gHszHH39sbrrpJisgateube3lJTgtyqVszZ492xQvXtxceumlpkWLFubEiRO+c/i6VatW9t4YScfxTUgqE7bCfnyG+2rfvr0VGCdPnjT9+vXznRfcFv0YMWKEdbJ51tdff71ZvHix+fXXX80DDzxg9912221m8+bNAddbt26dHdHn3rFzz549zalTpwLaHTlypI0E5M+f31x33XXm7bff9h0/d+6c6d69u+0vNua6iKFwaVFfffWVqVOnjr3e5Zdfbjp27GjvzRsJa9q0qRk3bpxtk3O6detm/vrrr7A2412in/Qd0V2rVi3TqVOngMgEkY2uXbva58Z7wntDndKqVavifR5CCCGEEJkicoHD6wUHjm3FihWmbt26AcfYh2Oemmk0XHv8+PF2Y4arli1bmhIlSpj+/ftbhxXnFSd12bJlvs8gPj744AOzZMkSc/z4ceto4yDiKAL/szL566+/bgUEwoHRahz6Nm3aWKcVQVO/fn3z7rvvmh9++ME89dRTAf06cOCAefDBB60Di6OL892nT5+L3g9CCHsvXbrU/P7779a+o0ePNi+++KI93rt3bzuijoNfuHBh288vv/wyojqSq666ygqVd955x5w/f96mTIUCAYMQYEYwvn788cetaMO2L730krU74oPRehfFIuqEKKFthAjPgM0biXr55Zft8xswYID58MMPTZcuXayYvfnmm82ECRPsPfKceI7Yky0UiJaGDRua6tWrm02bNpkjR46YJ5980l7P+z5/+umnVljwP+8AaWnYjWcbCtqjb4hEUrBol37ed999YW16+vRpK1jiiwCdPXvWbg7eQciVNdZkyxZr0gvxCbOktpkcbac3ZAs/skUgsocf2cKPbOFHtki8LSK1VUQrdOO89ujRI46wAJxrHLhBgwbZUexog4PNyLgXnD02R9u2ba0DDji5OIQ4wTibgNPPOV7O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l89cf/31ZvHixebXX381DzzwgN2H87158+aA6+GI33nnnfbesXPPnj3NqVOnAtodOXKkadeuncmfP7+57rrrzNtvv+07fu7cOevY019szHURQ+HSor766itTp04dez0iDTj/3Js3Eta0aVMzbtw42ybndOvWzfz1119hbcbzffTRR614oL+PPfaYadiwoVm7dm3AeYgJ7zt6xRVXxPsshBBCCCHSOxl2KJUUnvHjx9vt2WefNS1btjQlSpQw/fv3tw4rzitO6rJly3yfQXx88MEHZsmSJeb48ePW0WaRwDlz5tjj/P/888+b119/3TqYCIcOHTpYh56penFaETT169c37777rl2p/Kmnngro14EDB8yDDz5oHVgcXZzvPn36XPR+EEI4zUuXLjW///67FT6jR482L774oj3eu3dvG2nAwS9cuLDtJ6upR1JHctVVV1mh8s4775jz58/blKlQIGAQAiysyNePP/64FW3Y9qWXXrJ2R3x8/fXX1unnHog6IUpoGyHCM2DzRqJefvll+/wGDBhgPvzwQ9OlSxdz9913m5tvvtlMmDDB3iPPieeIPdlCgWjB6a9evbrZtGmTOXLkiHnyySft9RBwjk8//dQKC/7nHSAtDbvxbBMC7wGCjPvywur1RH4QQfQBEUSfQ3H27Fm7OXj/IFfWWJMtW6xJC8QnuFLiuql1/bSEbOFHtghE9vAjW/iRLfzIFom3RaS2SpfiAgebkXEvOKJsjrZt21oHHHByce5wgnE2Aaefc7ycOXPGzJo1y1xzzTX2+4kTJ5rGjRtbZ5eR5yFDhtivEQdwww03mF27dtloB+Ji7ty55sKFC2batGnWoWRk++DBg9Y5drz55pumZMmSth3AYWZ0fcyYMfHeM+3iEDOaDzjyq1atsuKCqMXMmTPt9evWrWuP46y7lKZIICpDu0ePHrViIxT33XefTQcCxAz3dvvtt5tmzZoF2P3w4cPWfjjXiBYXASlVqpQVCwgHPovNXLtu5XfaQLjg+GOrH3/80X6uVq1aVrAQuQgH9nDPFAEICMMmTZpYeyPCgKgX+xFR3DfPHNteTFxce+21ViDFxMTY6BLCxUHtCs+LPlMH8sILL9iIzc6dO33P0Au24ZxgBlW8YPLkOW/SAkTcUpMVK1ak6vXTErKFH9kiENnDj2zhR7bwI1sk3BanT582mUZckEqE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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "xgb.plot_importance(\n",
    "    gs_w2v_pipe.best_estimator_.named_steps['xgbregressor'],\n",
    "    max_num_features=20,\n",
    "    ax=ax,\n",
    "    importance_type='gain',\n",
    "    show_values=False,\n",
    "    height=0.6\n",
    ")\n",
    "\n",
    "# Get the current tick labels (e.g., 'f1', 'f2', etc.)\n",
    "tick_labels = [item.get_text() for item in ax.get_yticklabels()]\n",
    "\n",
    "# Replace tick labels with more descriptive labels\n",
    "for i, label in enumerate(tick_labels):\n",
    "    try:\n",
    "        feature_index = int(label[1:])  # Extract the feature index (e.g., 1 from 'f1')\n",
    "        tick_labels[i] = f\"Embedding Dimension {feature_index}\"  # Create new label\n",
    "    except ValueError:\n",
    "        pass  # Handle cases where label is not in the expected format\n",
    "\n",
    "# Set the updated tick labels on the y-axis\n",
    "ax.set_yticklabels(tick_labels)\n",
    "\n",
    "plt.title(\"Feature Importance - XGBoost (Word2Vec)\", fontsize=16)\n",
    "plt.xlabel(\"F score\", fontsize=12)\n",
    "plt.ylabel(\"Embedding Dimensions\", fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"xgb_word2vec.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "d27YNHgcfMeL"
   },
   "outputs": [],
   "source": [
    "# Predict\n",
    "predictions = gs_w2v_pipe.predict(X_test)\n",
    "predictions = list(map(round, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[154779  63683]\n",
      " [ 10623  27433]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.94      0.71      0.81    218462\n",
      "           1       0.30      0.72      0.42     38056\n",
      "\n",
      "    accuracy                           0.71    256518\n",
      "   macro avg       0.62      0.71      0.62    256518\n",
      "weighted avg       0.84      0.71      0.75    256518\n",
      "\n",
      "Specificity : 0.7084939257170584\n",
      "ROC-AUC : 0.7146763301067949\n"
     ]
    }
   ],
   "source": [
    "# evaluate\n",
    "cm = confusion_matrix(y_test, predictions)\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()\n",
    "print(confusion_matrix(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Specificity :\", tn/(fp+tn))\n",
    "print(\"ROC-AUC :\", roc_auc_score(y_test, predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2S2U299QVJdn"
   },
   "source": [
    "#### CBoW - All Sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# read/prep data\n",
    "data = pd.read_csv(\"../data/tokenized_reviews.csv\")\n",
    "data = data.dropna()\n",
    "data[\"quote\"] = data[\"quote\"].astype(int)\n",
    "data[\"tokenized_words\"] = data[\"tokenized_words\"].apply(lambda x: x.strip(\"[']\").replace(\"', '\",\" \"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1453600, 13), (1453600,))"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 85% train / 15% test\n",
    "X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=[\"popular\"]),\n",
    "                                                    data[\"popular\"],\n",
    "                                                    test_size = 0.15,\n",
    "                                                    stratify=data[\"popular\"],\n",
    "                                                    random_state = 229)\n",
    "\n",
    "X_train.shape, y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 4013108,
     "status": "ok",
     "timestamp": 1746040441650,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "WwYUhgyCVkxg",
    "outputId": "0bce11a9-ce38-448a-c5b7-186d4d62cb9e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "XGBOOST - RANDOM FOREST - CBoW\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107e1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105485d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104e95d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104bd9d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104b71d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1057a9d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1035c5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x103a1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106d05d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10561dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training completed in: 667.31 seconds\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# BAG OF WORDS\n",
    "print(\"\\n\\nXGBOOST - RANDOM FOREST - CBoW\")\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "# pipeline\n",
    "preprocessor = make_pipeline(\n",
    "    ColumnTransformer(remainder='passthrough',\n",
    "                      transformers=[('countvectorizer',\n",
    "                                     CountVectorizer(),'tokenized_words'), \n",
    "                                    ('standardscaler', StandardScaler(), numerical_cols) ]),\n",
    "    xgb.XGBRegressor(objective='binary:logistic',\n",
    "                     eval_metric='error',\n",
    "                     seed=229,\n",
    "                     n_jobs=-1))\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'xgbregressor__n_estimators': (100,1000),\n",
    "    'xgbregressor__max_depth': (4,6),\n",
    "    'xgbregressor__learning_rate': (0.1, 0.3)\n",
    "}\n",
    "\n",
    "# perform validation\n",
    "gs_bow_pipe = GridSearchCV(preprocessor,\n",
    "                           parameters,\n",
    "                           cv=ShuffleSplit(n_splits=1,\n",
    "                                           test_size=0.15,\n",
    "                                           random_state=229),n_jobs=-1)\n",
    "gs_bow_pipe.fit(X_train, y_train)\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print(f\"\\nTraining completed in: {total_time:.2f} seconds\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([107.47034478, 310.93231511, 156.06877899, 493.23196697,\n",
      "       115.25457311, 302.99355412, 147.07984376, 459.94970322]), 'std_fit_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'mean_score_time': array([17.1478672 , 15.10751987, 16.17884994, 15.67097807, 17.66402793,\n",
      "       14.57961392, 15.65037107, 14.84892988]), 'std_score_time': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'param_xgbregressor__learning_rate': masked_array(data=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=1e+20), 'param_xgbregressor__max_depth': masked_array(data=[4, 4, 6, 6, 4, 4, 6, 6],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'param_xgbregressor__n_estimators': masked_array(data=[100, 1000, 100, 1000, 100, 1000, 100, 1000],\n",
      "             mask=[False, False, False, False, False, False, False, False],\n",
      "       fill_value=999999), 'params': [{'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 4, 'xgbregressor__n_estimators': 1000}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 100}, {'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}], 'split0_test_score': array([0.15917826, 0.19115776, 0.17077911, 0.20175701, 0.17287731,\n",
      "       0.20361197, 0.18382955, 0.20764089]), 'mean_test_score': array([0.15917826, 0.19115776, 0.17077911, 0.20175701, 0.17287731,\n",
      "       0.20361197, 0.18382955, 0.20764089]), 'std_test_score': array([0., 0., 0., 0., 0., 0., 0., 0.]), 'rank_test_score': array([8, 4, 7, 3, 6, 2, 5, 1], dtype=int32)}\n",
      "{'xgbregressor__learning_rate': 0.3, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 1000}\n",
      "\n",
      "Best model saved as 'xgboost_bow_model_f.pkl'\n"
     ]
    }
   ],
   "source": [
    "# save the model\n",
    "print(gs_bow_pipe.cv_results_)\n",
    "print(gs_bow_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./xgboost_bow_model_f.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_bow_pipe.best_estimator_, f)\n",
    " \n",
    "print(\"\\nBest model saved as 'xgboost_bow_model_f.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 28,
     "status": "ok",
     "timestamp": 1746040489858,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "omFb2SgjVkxh",
    "outputId": "5fa1b512-e46e-4447-b2d6-e8127a3b1e76"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['ftc' 'supplied' 'youtube' 'br' 'divergent' 'katniss' 'tm' 'hype' 'fuck'\n",
      " 'policy']\n"
     ]
    }
   ],
   "source": [
    "# feature importance\n",
    "sorted_ind = gs_bow_pipe.best_estimator_.named_steps['xgbregressor'].feature_importances_.argsort()[::-1]\n",
    "feature_names = gs_bow_pipe.best_estimator_.named_steps['columntransformer'].transformers_[0][1].get_feature_names_out()\n",
    "num_bow_features = len(feature_names)\n",
    "bow_indices = sorted_ind[sorted_ind < num_bow_features]\n",
    "top_bow_features = np.take(feature_names, bow_indices[:10])\n",
    "\n",
    "print(top_bow_features)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 945
    },
    "executionInfo": {
     "elapsed": 1588,
     "status": "ok",
     "timestamp": 1746040491429,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "2HZGTtBNVkxh",
    "outputId": "b9467be4-19f6-41e7-a806-4418dc7d4f71"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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2L9/f/eZDf3b8ZpL628n9LNakKa+ah4u2b+D8kuf3dD361GyPW/ePPf/u+66y30Gc6PuB2q+npaWZlOnTs13E3Mt0wmxTowVl7wS8/w2LX/llVfcMdQ+w5XTGwsgJSUly7p3333XrVOz+1DZfxO8LgGKmVeu0NiF+63Q38EjjzxiRx99tPu712dJ35V5fcerqb362+vzps+dnqPft4cfftgyMjJybK/vt7zG8wjXlSG/v4N5KczrZl+uz41+g3RRR+MS6Ldbv4mhF3Cy07E7+eST3edSf2P6bnniiSf2W96Cfq/m97dc5yznnnuu+/7S90vVqlWtcePGduaZZ9rLL7+cZZ+7d++2Rx991L1HxVefCd3023DTTTfZzp078xwfR791r732mnt9fS40voK616xfvz64rbrr6PdRvzV6n3otrytHbnFQF6Ibb7zR/a7puOhv8pJLLsnzd60gXS9Cz6e+/fZbN0aDvn90LHVMddE2t/FA9N2kuLVu3dqVTd/f6p6kc7/czsNEf+tqZq6/97zOR1CCvCpvANFvXj59+vRAfHy8e07btm0DZ511VuD4448PlC9f3i1TE9TsLrnkErfuiCOOCPTp08c1Z+3cuXNwP9mbFj366KOBM888062rVKmSa5IcestvU2Wv2Wvoc0KX9+/fP3DYYYcFDjzwwMCpp54aOPvsswPnn39+cLtrrrnGbadyduzY0a3v1KmTayarJptPPPFEoCBya0rqlee0004LtGjRInDQQQe593/SSScFKlSo4NaNGDEi8M033wTq1q0baNCgQeCcc84JnHjiicGmo1OmTMn19UaNGuXeQ8uWLQP/+Mc/AieccELw2I8ZMybH89QkuVevXm69msWefPLJLmZ6XS2rVatW4OOPP861OZvKqv3rc3Heeee5Y7Z582YXB68ZdJcuXbLEdP78+cH99OzZ0z3/yCOPDPTr188dd6+Jq276DGbnHcOBAwcG2rRpE6hevXrglFNOccdUx9Nroqkmu9m9+eabbntto231HL3mMcccE0hMTMzx+fnxxx9d2bR9jRo13LE6/fTTg39LjRo1cu+3pP3555/u86My3HDDDYGmTZu6/0+ePDns9mvXrg0e008++aTQr7u/5uU65pUrV3bbvPbaaznWP/LII8HPo5pJ6zNz3HHHBcuWnJxcLJ/R9evXu+bxWt+8efPA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NwAAAAAAEULSHQWzZs2y6tWrR7sYAAAAAIAII+mOgnPPPde++uqraBcDAAAAABBhMdunOyMjw+Li4iw+Pv/XHfbt22fly5cv8mtXqFDB3QAAAAAA/laqku5GjRrZ6NGj3c3Ttm1bGzhwoN166602YcIEe+KJJ+yXX36xmjVr2llnnWX33nuv2y41NdVuuukme+aZZ2znzp3WunVru+OOO6x79+7BJt3a71NPPWU33HCDq2n+5ptv3GvmZsiQIW5fxxxzjD3wwAOWlJRk3377rX3//fd2zTXX2BtvvOGS9q5du9qMGTPcvrTs1FNPtZ9//jlLE/KrrrrK1q9fb2+99VawLNq35+WXX3bv7/PPP7f69evb4MGD3fspV66cjR071r788ktbuHCh23b69Ol29dVX22uvvWZ9+/Z1y5o0aeLe16WXXmrLli2z6667zj777DNLTEy0Vq1a2dy5c+3QQw8N+z517HTzpKSkuPuk+IAlJAQKHU+ULopn6D38g9j6F7H1L2LrD2lpabkuC7cOZRuxjd3YphUx5qUq6c7Liy++aNOmTbN58+a5JFJJ7bp164LrR4wY4RJWrVfSOn/+fJeQKtFt2rSp22bv3r0uEX/sscdc0n7QQQft93WXLl1qVatWtSVLlgQPeJ8+faxz5862fPlylxRPmjTJvdann35qPXv2dMm2ynvJJZcEa9WfffZZ+9e//hX2NbSfCy+80F1AUAK/adMmGzZsmFuniw3dunVzZdZ+EhIS7J133rFatWq55Fqv++OPP7rn6AJDenq6u0gxdOhQdwFCtfMffvihq9XPzeTJk13Cn93N7TKtYsWM/R4jlC0TO2RGuwiIEGLrX8TWv4ht2bZo0aJc13nnjvAfYht7sd27d29sJN1btmyxunXrWq9evVztbcOGDa1jx47BdTNnznT3SrhFtcOvv/66W3777bcHE+YHH3zQjjrqqHy/bqVKlVzC6zUrnzNnjmVmZrplXiKr11CirST4pJNOsn/84x+uZtlLupW4q1b7zDPPDPsaSnhVS63abWncuLFNnDjR1VYr6VYivnv3bluzZo0dffTR9u6779q1115rCxYscNvrdQ8++GBX2/3777/brl27bMCAAXb44Ye79S1atMjzPY4bN87GjBmTpaa7QYMGNmlNvKUnJuT7WKF0U22KTu7Gr4631Eymp/ETYutf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nJwcXLZ161Y7+eSTXY15vXr1bOrUqXbWWWcF16tc11xzjb3xxhsWHx9vXbt2tRkzZrj95yY1NdXdPCkpKe4+KT5gCQmBYjm2iD7FM/Qe/kFs/StWYpuWlmaxxnvPsfje/Y7Y+hexjd3YphUx5iTdIc4++2wbNWqUvfLKK+7/sm3bNnv11VddEjt//ny76qqrbPr06darVy9buHChXXTRRXbIIYcEk+m8NGjQwF588UU788wzbePGjVa1alWrUKFCgco4fvx4mzJlikukdRHgH//4h61fv95atGjhPgx9+vSxzp072/Lly61cuXI2adIk69u3r3366adWvnz5sPucPHmyTZgwIcfym9tlWsWKGQUqH0q/iR0yo10ERAix9S+/x3bRokUWq5YsWRLtIiBCiK1/EdvYi+3evXuLtF+S7hBKgAcNGmQzZ84MJt1z5syxhg0bWvfu3e3444+3IUOG2BVXXOHWjRkzxj744ANX852fpDshIcFq1Kjh/n/QQQdZ9erVC1xGlevSSy91/584caL7YNx333324IMP2rPPPmuZmZn22GOPuX7jovei11HN+EknnRR2n+PGjXPvJbSmWxcIJq2Jt/TEhAKXEaWTasp04j5+dbylZvp36qFYRGz9K1ZiuyG5j8UaXSjXb3jv3r0tMTEx2sVBMSK2/kVsYze2KX+3BC4sku5shg4dasccc4z9+OOPdvDBB7um30q0lcR+8cUXNmzYsCzbd+nSxdU6lxTVYmd/vHbtWvf/devW2TfffOOawof666+/bNOmTbnuU03pQ5vTe3SCl+7jeWFjleLq5/l+Yxmx9S+/xzaWT1713mP5/fsZsfUvYht7sU0sYrxJurNp166dHXXUUa5/t2qGP/vsM9e8PD/Uh1oCgUCB2//ruaHPK8hzPXv27LGjjz7ann766RzrateuXaB9AQAAAACKjtHLw1DzbdVwq2m2+m6rqbWo3/SKFSuybKvHLVu2zJLYarAzj1cL7fH6VWdkZO0rreeGPk9NGL799tscZVNz9uyPVS5p3769ff31167pepMmTbLcqlWrVsijAQAAAAAoLJLuMNSv+4cffrBHH33ULr744uDya6+91iXjGsFcye0999xjL730ko0dOzbYJ/zYY491A52pKfo777zjRjcPdeihh7qm6hqE7ddff3W109KjRw83MJoGQNPAaBrV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rVq1almepybs+amlHzNmTJaabg0AN2lNvKUnJuT7PaF0Uy3oxA6ZNn51vKVmMpeznxDbyNuQ3Ccqr6sr60uWLLHevXtbYmJiVMqAyCC2/kVs/YvYxm5sU/5uCVxYJN3FoEKFCgV+Tmgw1ffa6xeeH3v27HH3as7eqVOnHAO3hVJz8/1RH3TdstPJe3oGJ/B+o7imEldfIraRE+2TK71+tMuAyCC2/kVs/YvYxl5sE4sYbwZSKwZNmzZ1ibcGRysOGhxt1apVWZZ98MEHwf9rgLX69evbf//7X9fMPPSmJuUAAAAAgNKBmu5ioH7Y119/vV133XVWvnx569Kli/3666/22WefFajJuUd9sIcMGeKahmtfGjRN+wodSE19sEeNGuWak/ft29c1b1+9erXt2LEjS1NxAAAAAED0kHQXk/Hjx7u5uDWi+E8//eT6aWfvl51f5557ruuzrSReg6edeeaZdvnll9vixYuD22jANg1yduedd9q1117rmpEfeeSRNnr06GJ8VwAAAACAomD0coQdKEA16Nu3b7eaNWtGuzgoxgEiNLp9v3796IfkM8TWv4itfxFb/yK2/kVsYze2KX/nR4UdvZw+3QAAAAAARAhJNwAAAAAAEULSDQAAAABAhJB0AwAAAAAQISTdAAAAAABECEk3AAAAAAARQtINAAAAAECEkHQDAAAAABAh5SK1Y5R9nSYvtfRylaJdDBSTpISATe1o1jp5saVmxEW7OChGxDanzVP6R7sIAAAADjXdAAAAAABECEk3AAAAAAARQtIdRZs3b7a4uDhbu3ZttIsCAAAAAIgAku4o2bdvX7SLAAAAAACIMAZS+1tmZqbddddd9sgjj9j3339vderUscsuu8xuuukmW79+vV111VW2cuVKq1ixop155pl2zz33WOXKld1zu3fvbm3btrXp06cH9zdw4ECrXr26zZo1yz1u1KiRXXLJJfb111/bggUL7IwzzrAnn3zSrWvXrp2779atmy1btsz9/7HHHrO7777bvv32W/fcUaNG2RVXXBFM2MeMGWMvvvii7dixw5V1+PDhNm7cOLv44ott27ZttnDhwmBZ0tLS7OCDD7bJkye7MmSXmprqbp6UlBR3nxQfsISEQESON0qe4hl6D/8gtjnpe89P78Mv7wf/j9j6F7H1L2Ibu7FNK2LMSbr/poT10UcftWnT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XXHFFXbEEUe4Uc01pVko7Vf90HXhAAAAAAAQPb4evTySNBK6pvLyRi8vTVST3rNnTzcvuOYdLygNFFCtWjU3FZlGboc/aIyBRYsWuS4I9EPyF2LrX8TWv4itfxFb/yK2sRvblL/zI0Yvhxup/Ndff3VTgGnE8sIk3AAAAACA4hPzzcv95JlnnrFDDz3Udu7caVOnTo12cQAAAAAg5lHTXUgawKy00QBqugEAAAAASgdqugEAAAAAiBCSbgAAAAAAIoSkGwAAAACACCHpBgAAAAAgQki6AQAAAACIEEYvR646TV5q6eUqRbsYKCZJCQGb2tGsdfJiS82Ii3ZxUIxiIbabp/SPdhEAAAAKhZpuAAAAAAAihKTbB/OFx8XF2c6dO6NdFAAAAABANiTdAAAAAABECEl3BKWlpUW7CAAAAACAKPLFQGq7d++24cOH24IFC6xq1ap23XXX2csvv2xt27a16dOnW2pqqt100032zDPPuGbYrVu3tjvuuMO6d+/unj9r1iwbPXq0Pfvss+7++++/t+OPP95mzpxp9erVC77OY489Znfffbd9++231qhRIxs1apRdccUVbt3mzZvtsMMOs3nz5tmDDz5oq1atsn//+9924YUX2qRJk+yRRx6xX3/91Vq0aGFTpkyxvn37BpuHn3jiibZjxw6rXr26W7Z27Vpr165d8HW+++47GzFihL333nu2b98+t+zOO++0fv36ZTkOf/zxhyvvE088YWeddVZwuY7L+eefbz///LNVqVIlx/HT8dHNk5KS4u6T4gOWkBAo5mghWhTP0Hv4RyzENlYvYnrvO1bfv58RW/8itv5FbGM3tmlFjLkvku4xY8bYihUr7JVXXrE6derYLbfcYp988olLukUJ6+eff+4S4vr169v8+fNd0rt+/Xpr2rSp22bv3r1211132ezZsy0+Pt7++c9/2tixY+3pp59263Wv/d5///0uIV6zZo0NHTrUKlWqZIMHDw6W5YYbbnCJubY54IADbMaMGe7xww8/7JYpIT711FPts88+C772/lx55ZUu2X733Xfd6+m9VK5cOcd2WvePf/zDXSwITbq9x+ESbpk8ebJNmDAhx/Kb22VaxYoZ+Sojyo6JHTKjXQREiJ9ju2jRIotlS5YsiXYRECHE1r+IrX8R29iL7d69e2M76VYt95NPPmlz5861nj17BpNMJdeyZcsW91j33jIl06+//rpbfvvttwevXqhm+vDDDw8m6rfddlvwdW699VaXPJ9xxhnusWq1lfwqmQ5NulVT7m0jSuSvv/56lwyLatjffvttVwP/wAMP5Os9quxnnnmmHXnkke5x48aNc9320ksvteOOO862bt3qar23bdvmTlbffPPNXJ8zbtw4d+EitKa7QYMGNmlNvKUnJuSrjCj9VAuqpGz86nhLzfTntFKxKhZiuyG5j8Ui/TbpBKB3796WmJgY7eKgGBFb/yK2/kVsYze2KX+3BI7ZpPu///2vO0gdO3YMLqtWrZo1b97c/V+12RkZGdasWbMsz1Nz6po1awYfV6xYMZhwi5ewes22N23aZJdccomr3fakp6e71wrVoUOHLMH56aefrEuXLlm20eN169bl+z2qGfvll19ub7zxhvXq1csl4G3atAm7rY5Dq1at3IUI1brPmTPHDj30UDvhhBNy3X9SUpK7ZaeT93SfzvkbyxRXv87lHOv8HNtYP7nR+4/1Y+BXxNa/iK1/EdvYi21iEeNd5pPu/dmzZ48lJCTYxx9/7O5DhTbRzn4gNQ1XIBAI7kMeffRR69SpU5btsu9TTbwLQk3ZxXutcH0GVHvdp08fe/XVV13irebgqnUfOXJk2H1qe9WiK+lWbf5FF13k3g8AAAAAoGSV+dHL1dRaCfNHH30UXLZr1y776quv3P/Vj1o13aq1btKkSZZb3bp18/Ua6ieupumqVc++DzUzz40GddPz1N88lB63bNnS/b927druXs3BPRpILTs199ZgcS+99JJdc8017gJAbtQfXYOv3Xvvva4JfGjzdwAAAABAySnzNd0aHExJ5bXXXms1atSwgw46yPW/Vg2yanfVrFwjd2sUcW+AM40ivnTpUtdEu3///vl6HQ00pmbeak6uQdjUPH316tVu1PHQ/tDZqVwqj5qua2A31TwrqfYGaFPiroQ6OTnZ/vWvf7mLBSpnKPUTP/nkk9170eupT7hGQc/NgQce6PqV67VPOukkO+SQQ/J9PAEAAAAAxafM13TLPffcY507d7YBAwa4Ps/qM62kVKOHixJdJd2qIVZf74EDB7qa8YYNG+b7NdRkW1OGaV8a0Kxbt25uqrG8arpFibqScr22nqcB3DTKujdyuWrpNZXZl19+6S4CaKA1TTEWSjX1GsFc70kJv5JvTUuWF/U/14jnF198cb7fIwAAAACgeMUFQjsT+4QGPjv44INdjbGSz1ikqc+uvvpqN5Bb+fLlC/RcDQCnGv3t27dnGWwOZZvGCtBI9prfncE//IXY+hex9S9i61/E1r+IbezGNuXv/EjdmNWFOOaal4vmzFZNsUbu1oHwpvo67bTTLNZoDjn1D58yZYpddtllBU64AQAAAADFxxfNy735sI866ijXvFw13cuXL7datWpZrJk6daodccQRbpA4zb8NAAAAAIgeX9R0a3A0TQkGcwOy6QYAAAAAiD7f1HQDAAAAAFDakHQDAAAAABAhJN0AAAAAAEQISTcAAAAAABFC0g0AAAAAQIT4YvRyREanyUstvVylaBcDxSQpIWBTO5q1Tl5sqRlx0S4OYjC2m6f0j3YRAAAAShw13T7TqFEjmz59evBxXFycLViwIKplAgAAAIBYRdINAAAAAECEkHRH0L59+6JdBAAAAABAFJF0F6Pu3bvbiBEjbPTo0VarVi3r06ePvfPOO9axY0dLSkqyevXq2Q033GDp6elu+4ULF1r16tUtIyPDPV67dq1rDq5tPJdeeqn985//DD5+7733rGvXrlahQgVr0KCBjRo1yv74448ovFsAAAAAwP4wkFoxe/LJJ+3yyy+3FStW2M8//2z9+vWzIUOG2FNPPWVffvmlDR061A444ABLTk52yfPu3bttzZo11qFDB5egK1lftmxZcH9adv3117v/b9q0yfr27WuTJk2yJ554wn799VeX5Os2c+bMQpc5NTXV3TwpKSnuPik+YAkJgSIdD5QeimfoPfyjrMQ2LS0t2kUoc7xjxrHzH2LrX8TWv4ht7MY2rYgxjwsEAqX7LK2M1XQrYf3kk0/c45tuuslefPFF++KLL1wNtjz44IMuid61a5fFx8fb0Ucfbeedd56NHTvWTj/9dDvmmGNswoQJ9ttvv7ltDjnkEPvqq6+sadOmrtY7ISHBHn744Sw13926dXO13UrmNZCaatp1E73u/PnzbeDAgbmWWxcA9JrZzZ071ypWrBiBIwUAAAAAZcPevXtt0KBBLj+rWrVqgZ9PTXcxUxLtUbLduXPnYMItXbp0sT179tgPP/xgDRs2dAmzaravueYaW758uU2ePNmee+45l0z//vvvVr9+fZdwy7p16+zTTz+1p59+Org/XTPJzMy0b7/91lq0aFGo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b968HHynAQAAAACWrl7u5HA4xJdoeB0+fLjExMTId999Z4Ztt2zZUjp06CB33XWXFChQQL788kspUqSIvPHGG9KuXTvZt2+fCcINGzY0YbdJkyby448/muC6fft2OXfunBQsWFDWrVtnQqx65ZVX5LPPPpOPPvrIXB5NQ7VuGfUya+9/ixYt5Ntvv5Xg4GCZOHGidO7cWX744QcZMWKEmQMfFxcns2fPNo/R9mzevFmaNWtm5snXqVNH8uXLZ+7TAD1u3Dh59dVXTSDXNvbv31/Cw8PloYcecj2vzqmfOnWqOUaDd3oSEhLM5qRtUKGBDgkK8q3aIuu0nu5f4Zv0d0VWH5OVx8K3UVv7orb2RW3ti9r6b20Ts1nzHAndvkaHvj/zzDPm++rVq5tgumbNGhO2NcQeO3bMdSmzKVOmmMXOFi1aJAMGDDA9yRq6NQTrVw3qP//8s6xfv94EZN2nvejq8OHD5vw33nijCefa050R7XHWnnPtnXZe01zDtfZ66zl1Hry2T8Ov9s47aY+8KlGihMd+fX0apnVov9Ie8d27d5sPEdxD99ChQ13HZER758ePH59m/5hGekm45Ct815FXTGiS4u0m4BKWL1+e5cfqNB/YE7W1L2prX9TWvqit/9U2Pj4+W+e1beh2p8O/NWjrMHLtsdYA6+7ChQty8OBB8732Yr/zzjuSnJxserU1DGvY1WCs5z1w4IBriLf2oGsor1mzpgnk3bp1M8enR59bH5t62P0///zjeu4rdf78efOYvn37mt5t98u0ae+9O+2xv5JV6HVkgHtPtw6hn7g9UJJCgjLVNvgu7eHWwD12a6AkpHCdbl+1K7JTph+jn77qHwn9fcS6DfZCbe2L2toXtbUvauu/tY37dyRwroZunS+sq5M750ErXURNe20zup53bkr9RmnPsvYya+DWAK4BOjVn23W19bNnz5rX980338jzzz9vQrfOZW/QoIGUK1fO9G47X9ehQ4fMUHUd/q3ztdu3b296zVPT577uuuvSnVft7M2+Unou9dZbb0nz5s097tM56e50uPnlaK+/s+ffnQazpGTCmd1oXROoq8/Kzh9xfSz/CLAnamtf1Na+qK19UVv/q21INuudpdA9duxYs7l79NFH053rrYFXe419gYbko0ePmvnUlStXTvcYDd/ao61D0vXN1QXKrrrqKunZs6d8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u69ChQry559/St26ddM9R8+ePWXfvn2Zet7KlSvL9OnTc+Q1AAAAAAByHnO6c8DBgwela9euUqlSpUvuK1OmTIbnKFCggNkAAAAAAPZBT/cVWLFihdx4441m+HeJEiWkW7duJlQrHSK+bds2efbZZ833kZGR6e5LPbw8tdTDy/X8t956q5QuXVoKFiwoTZs2ldWrV7vub9u2rfz6668ybNgwc17dnNavXy+tWrUyIV572J944gk5f/68pe8RAAAAACAterqvgAbW4cOHS/369eXcuXMybtw4ue2220yA1iHj7du3l86dO8uIESNMQB44cGCafSdOnMjUc+rz3HLLLfLcc89JaGiovPfee9K9e3czXL1ixYryySefSIMGDWTAgAHSv39/j7Cuzztx4kR599135fjx4/LYY4+Zbfbs2ek+V0JCgtmc4uLizNfQQIcEBTmy/L7Bt2g93b/COxITEy07pxXnhndRW/uitvZFbe2L2vpvbROzWXNC9xW44447PG5rmC1VqpTs3r3bzNEODg42wdo5fFy/T70vs6FbA7VuThMmTJBPP/1UPvvsMxOgixcvLkFBQVKoUCGPYetRUVFy3333ydChQ83t6tWryyuvvCJt2rSR119/3Szslpo+Zvz48Wn2j2mUImFhyZlqN3zfhCYp3m6CX1u+fLll5161apVl54Z3UVv7orb2RW3ti9r6X23j4+OzdV5C9xXYv3+/6d2OiYkx4Tkl5X+h5fDhwxkujJZd2tOtw9KXLVtmetOTkpLkwoUL5jkvZefOnfLDDz/IvHnzXPscDodp86FDh6RWrVppHhMREWF68t17unVY+sTtgZIUEpTDrwzeoj3cGrjHbg2UhBSu0+0tuyI75fg59dNX/SPRoUMHCQkJyfHzw3uorX1RW/uitvZFbf23tnH/jgTOKkL3FdBh3bog2ltvvSXlypUzAVbD9sWLFy17Th2WroWfMmWKVKtWzczPvvPOOy/7nBrWH3nkETOPOzUdlp4eHb6uW2oazJKSCWd2o3VNoK5eY+UfaT03/wiwJ2prX9TWvqitfVFb/6ttSDbrTei+jJMnT5p51Bq4dXEy50JlVtuwYYP07t3bzB13hmldjM1dvnz5JDnZc/h348aNzbB3DeoAAAAAAO9i9fLLKFasmFmx/M0335QDBw7I119/7TEU2yo6F1sXS9PF2nTIeK9evVzD2t2v0/3NN9/I77//7poz/tRTT8nGjRvNvG99rA6NX7p0qbkNAAAAAMhdhO7LCAwMlIULF5pLgOmQcr1E14svvmj587700ksm8N9www1meHunTp1ML7Y7vSSZ9n5XrVrVLOymdIX1devWyb59+0zPfKNGjcx8dB0WDwAAAADIXQwvvwJ6+S8dsu1OFydzSu/a26n3aa+0+2P0Otvut3UouW7ux2uvurvBgwd73L7++utNL3hqek3vr776SrIrJqKd6eWHfRaI0JWzdSEv5iEBAAAAuYOebgAAAAAALELoBgAAAADAIoRuAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALELoBgAAAADAIsFWnRh5X/OoNZIUHO7tZiCHhAY5ZHIzkbqRKyUhOcDbzfEbsZO6ersJAAAA8CJ6ugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKHbZhITE73dBAAAAADAv1hIzcetWLFCJk6cKLt27ZKgoCBp0aKFvPzyy1K1alWJjY2VKlWqyMKFC2XmzJkSExMjs2bNkt69e8u7774rU6dOlQMHDkjx4sXljjvukFdffTXd50hISDCbU1xcnPkaGuiQoCBHrr1WWEvr6f4V9vkgzPkcfOhmP9TWvqitfVFb+6K2/lvb7NY8wOFw8C9wH7Z48WIJCAiQ+vXry7lz52TcuHEmbO/YsUMOHz5sQnflypVNwG7UqJHkz59flixZIsOHD5dJkyZJly5d5MyZM7JhwwYZOnRous8RGRkp48ePT7N//vz5EhYWlguvEgAAAAB8U3x8vPTq1cvkqsKFC2f68YTuPObEiRNSqlQp+fHHH6VgwYImdE+fPl2GDBniOqZ8+fLSp08f00N+JdLr6a5QoYLUHrlQkkK4ZJhdaA/3hCYpMnZroCSkcMmw3LIrspPlz6Gfvq5atUo6dOggISEhlj8fcg+1tS9qa1/U1r6orf/WNi4uTkqWLJnl0M3wch+3f/9+07utQ8c1cKekpJj92stdu3Zt832TJk1cxx87dkz++OMPadeu3RU/R2hoqNlS02CWxPWcbUfrynW6c09u/lHW5+IfAfZEbe2L2toXtbUvaut/tQ3JZr0J3T6ue/fuUqlSJXnrrbekXLlyJnTXrVtXLl686DomPPz/e6MLFCjgpZYCAAAAAFJj9XIfdvLkSdm7d6+MGTPG9FzXqlVLTp06dcnHFCpUyMzxXrNmTa61EwAAAACQPnq6fVixYsWkRIkS8uabb0rZsmXNkPJRo0Zd9nG6MNrAgQPlqquuMgupnT171iyk9vjjj+dKuwEAAAAA/0NPtw8LDAw0lwPbtm2bGVI+bNgwefHFFy/7uIceesgsrqaXEatTp45069bNzA0HAAAAAOQuerp9XPv27WX37t0e+9wXnM9o8flHHnnEbNkRE9HO9LTDPqsyLl++3KymzeIfAAAAQO6gpxsAAAAAAIsQugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLELoBAAAAALAIoRsAAAAAAIsQugEAAAAAsEiwVSdG3tc8ao0kBYd7uxnIIaFBDpncTKRu5EpJSA7wdnNsL3ZSV283AQAAAD6Anu4c0LZtWxk6dGiOnjM2NlYCAgJkx44dGR4THR1tjjl9+nSOPjcAAAAAIGcQugEAAAAAsAihGwAAAAAAixC6c0hSUpI89thjUqRIESlZsqSMHTtWHA6HuU+HgC9ZssTj+KJFi8qcOXNctzdv3iyNGjWS/PnzS5MmTWT79u1pnmP58uVSo0YNKVCggNx0001mCLrT+fPnpXDhwrJo0SKPx+jzhoeHy9mzZy141QAAAACAS2EhtRwyd+5c6du3rwnPW7dulQEDBkjFihWlf//+l33suXPnpFu3btKhQwf54IMP5NChQzJkyBCPY44cOSK33367DB482Jxbn+O///2v634N1vfcc4/Mnj1b7rzzTtd+5+1ChQpl+PwJCQlmc4qLizNfQwMdEhT0vw8OkPdpPd2/wlqJiYm5/ly5+ZzIHdTWvqitfVFb+6K2/lvbxGzWnNCdQypUqCDTpk0zvdo1a9aUH3/80dy+ktA9f/58SUlJkXfeecf0dNepU0d+++03GTRokOuY119/XapWrSpTp041t53P8cILL7iO6devn9xwww3y559/StmyZeXYsWOmd3z16tWXfP6oqCgZP358mv1jGqVIWFhyJt8J+LoJTVK83QS/oD97uW3VqlW5/pzIHdTWvqitfVFb+6K2/lfb+Pj4bJ2X0J1Drr/+ehO4nVq0aGECcnLy5UPrnj17pH79+iZwuz8+9THNmzf32Jf6mGbNmpnArr3uo0aNMr3mlSpVktatW1/y+SMiImT48OEePd36IcLE7YGSFBJ02fYjb9Aebg3cY7cGSkIKlwyz2q7ITrn2XPrpq/6R0NEyISEhufa8sB61tS9qa1/U1r6orf/WNu7fkcBZRejOBRrGnfO7nawalqK93a+99poJ3Tq0vE+fPh4fBqQnNDTUbKlpMEvies62o3XlOt3W88YfY31O/hFgT9TWvqitfVFb+6K2/lfbkGzWm4XUckhMTIzH7U2bNkn16tUlKChISpUqZYZ8O+3fv99jiEKtWrXkhx9+kH/++cfj8e70GJ0vnvo5Urv//vvl119/lVdeeUV2794tDz30UI68PgAAAABA5hG6c8jhw4fNEO29e/fKggULZMaMGa7F0G6++WZ59dVXzYrkugDawIEDPT4t6dWrl+mN1vnfGpR1LuiUKVM8zq+P0bA+cuRI8xw6D9x99XOnYsWKmQXX9LiOHTvK1VdfnQuvHgAAAACQHkJ3DnnwwQflwoULZl61rjCugVtXGVc6t1vnSLdq1coE7BEjRkhYWJjrsQULFpTPP//cLIymlw0bPXq0xwJpSldCX7x4sbkEWIMGDWTWrFny/PPPp9sWXUX94sWL8vDDD1v8qgEAAAAAl8Kc7hwQHR3tscp4auXKlZOVK1d67Dt9+nSahdh27NjhsS/1PHC9rJhu7nTOdmq///67lChRQm699VbJjpiIduY8sAddR0BHUegCX8xDAgAAAHIHodtGdJ64zh2fNGmSPPLII5IvXz5vNwkAAAAA/BrDy21k8uTJcu2110qZMmXMZcAAAAAAAN5F6LaRyMhIM4R4zZo1Zp44AAAAAMC7CN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhOt0I0PNo9ZIUnC4t5uBHBIa5JDJzUTqRq6UhOQAbzfHdmIndfV2EwAAAOCD6OkGAAAAAMAihG4AAAAAACxC6PYRc+bMkaJFi6bZX7lyZZk+fbpX2gQAAAAAyB5Ct5+4ePGit5sAAAAAAH6H0J1D2rZtK4899pjZihQpIiVLlpSxY8eKw+Ew9586dUoefPBBKVasmISFhUmXLl1k//795r7o6Gjp06ePnDlzRgICAswWGRlpzvnrr7/KsGHDXPud1q9fL61atZICBQpIhQoV5IknnpDz58979JBPmDDBPGfhwoVlwIABXnhXAAAAAMC/sXp5Dpo7d6707dtXNm/eLFu3bjVBt2LFitK/f3/p3bu3CdmfffaZCcFPPfWU3HLLLbJ792654YYbzBDycePGyd69e825ChYsaIJ0gwYNzHn0HE4HDx6Uzp07y8SJE+Xdd9+V48ePuwL/7NmzXcdNmTLFnPOZZ565ZLsTEhLM5hQXF2e+hgY6JCjofx8aIO/Terp/Rc5KTEz0+nN7sw2wBrW1L2prX9TWvqit/9Y2MZs1D3A4u2KRLdorfezYMfnpp59cPdKjRo0yIXvp0qVSo0YN2bBhgwnY6uTJk6aHWoP6XXfdZeZ0Dx06VE6fPu1xXu2x1v26OfXr10+CgoLkjTfe8Oj5btOmjentzp8/v3lco0aN5NNPP71s27VXffz48Wn2z58/3/TKAwAAAIC/io+Pl169epmRydqBmln0dOeg66+/3mMIeIsWLWTq1KmmNzs4OFiaN2/uuq9EiRJSs2ZN2bNnT6afZ+fOnfLDDz/IvHnzXPv0s5OUlBQ5dOiQ1KpVy+xr0qTJFZ0vIiJChg8f7tHTrR8ITNweKEkhQZluH3yT9nBPaJIiY7cGSkIK1+nOabsiO3ntufXT11WrVkmHDh0kJCTEa+1AzqO29kVt7Yva2he19d/axv07EjirCN150Llz5+SRRx4xw89T0+HsTuHh4Vd0vtDQULOlpsEsKZlwZjda1wTqmuN84Y+vtsEX2oGcR23ti9raF7W1L2rrf7UNyWa9Cd05KCYmxuP2pk2bpHr16lK7dm1JSkoy97sPL9f523qfypcvnyQnJ6c5Z3r7GzdubHrPq1WrZunrAQAAAABkD6uX56DDhw+bYdoaphcsWCAzZsyQIUOGmOB96623msXQdO61Dg+///77pXz58ma/0jnY2oO9Zs0aOXHihJk34Nz/zTffyO+//272K12EbePGjWbhtB07dpgF2nTeuN4GAAAAAPgOQncO0stzXbhwQZo1ayaDBw82gdt5qS5dVfy6666Tbt26mbneOgd7+fLlrqEK2gM+cOBA6dmzp5QqVUomT55s9j/77LMSGxsrVatWNftV/fr1Zd26dbJv3z5z2TBdME1XKS9XrpwXXz0AAAAAIDWGl+cgDdB66a/XX389zX16fe733nvvko/Xx6V+rC7Opj3jqTVt2lS++uqrDM+lQT27YiLamQXfYJ8FIvSDHl3wi3lIAAAAQO6gpxsAAAAAAIsQugEAAAAAsAjDy3NIdHS0t5sAAAAAAPAx9HQDAAAAAGARQjcAAAAAABYhdAMAAAAAYBFCNwAAAAAAFiF0AwAAAABgEVYvR4aaR62RpOBwbzcDOSQ0yCGTm4nUjVwpCckB3m6ObcRO6urtJgAAAMCH0dMNAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWISF1DKwYsUKmThxouzatUuCgoKkRYsW8vLLL0vVqlUlNjZWqlSpIh9++KHMmDFDtm7dKnXr1pV58+bJmTNnZNCgQfLzzz9Lq1at5L333pNSpUqZc27ZskWefvpp2b59uyQmJkrDhg1l2rRp0rhxY3P/nDlzpE+fPmna8swzz0hkZKSkpKSYNr355pty/PhxqVWrlkyaNEk6d+5sjnO2a/HixaZdMTExUr16dZk1a5Zpf0YSEhLM5hQXF2e+hgY6JCjIkePvLbxD6+n+FTlDf5Z9pQ2+0BbkLGprX9TWvqitfVFb/61tYjZrHuBwOPgXeDo0uAYEBEj9+vXl3LlzMm7cOBNqd+zYIYcPHzbh9tprr5Xp06dLxYoV5eGHHzbFKFSokAnGYWFhcvfdd0v79u3l9ddfN+f8+uuv5Y8//pAmTZqIvu1Tp06VL774Qvbv328ed+HCBRPanaKjo+WBBx6Q5cuXS4cOHUxA1/D9xhtvSKNGjeTdd981+3766ScTrp2hW9s1ZcoUs2/06NEm7B84cECCg9P/jEXPOX78+DT758+fb14HAAAAAPir+Ph46dWrl8lqhQsXzvTjCd1X6MSJE6bH+scff5SCBQuacPv2229L3759zf0LFy6Ue++9V9asWSM333yz2ae90Np7rb3e6dGe66JFi5pw261bN4/7Dh48KM2aNZNRo0bJyJEjzb7y5cvL4MGDTW+5kx7TtGlTee2111yh271du3fvljp16siePXtMGL/Snu4KFSpI7ZELJSmES4bZhfZwT2iSImO3BkpCCpcMyym7Ijt5uwnmA79Vq1aZD+dCQkK83RzkIGprX9TWvqitfVFb/61tXFyclCxZMsuhm+HlGdDeZ+3d1iHaGrg1ICvt5a5du7b5XnvBnUqXLm2+1qtXz2PfsWPHXLf/+usvGTNmjOnB1v3JycnmUxM9pzstpobwrl27ugK3Flp7yVu2bOlxrN7euXOnxz73dpUtW9Z81efLKHSHhoaaLTUNZklcz9l2tK5cpzvn+NIfXW2LL7UHOYfa2he1tS9qa1/U1v9qG5LNehO6M9C9e3epVKmSvPXWW1KuXDkTunXe9sWLF9N983Uoenr7nGFdPfTQQ3Ly5EkzN1zPrUFX51q7n1ODeM+ePc0nKDp3OyvSa5d7OwAAAAAAuYPQnQ4Nxnv37jWBWxdDU+vXr8/2eTds2CAzZ86UW265xdw+cuSI6UV3N2zYMDOEXRdny58/v2u/hnAN/3qONm3aeJxTh5gDAAAAAHwPoTsdxYoVkxIlSpieZh2ercO/dW51dunCZu+//75ZSE2Hi+vQ8QIFCrjunz17tgnln376qemhPnr0qNmvc8h10+N1JXNdQV1XPtfjdWE3XTUdAAAAAOB7uE53OgIDA83CaNu2bTNDyrX3+cUXX8z2ed955x05deqUuUSYrkr+xBNPyFVXXeW6f926dWZ4+X/+8x8T9p2brkSu9Pjhw4fLf//7XzN3XC9r9tlnn5kwDwAAAADwPaxejjS0F75IkSJm6Lv2+MM+qzLq5ed0egOLf9gLtbUvamtf1Na+qK19UVv/rW3cv/koq6uX09MNAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFgq06MfK+5lFrJCk43NvNQA4JDXLI5GYidSNXSkJygLebYxuxk7p6uwkAAADwYfR0AwAAAABgEUJ3FrRt21aGDh2a4f2VK1eW6dOn52qbAAAAAAC+h+HlFtiyZYuEhzMsGwAAAAD8HT3dFihVqpSEhYVZdv7ExMQ0+y5evGjZ8wEAAAAAsobQnUVJSUny2GOPSZEiRaRkyZIyduxYcTgcaYaXz5kzRwICAtJskZGRrl7xDh06mHPoudq0aSPff/+9x3Pp8a+//rr85z//MT3ozz33nHl8w4YN5e2335YqVapI/vz5zbGHDx+WW2+9VQoWLCiFCxeWu+++W/76669cf38AAAAAAAwvz7K5c+dK3759ZfPmzbJ161YZMGCAVKxYUfr37+9xXM+ePaVz586u29HR0fLAAw9Iy5Ytze2zZ8/KQw89JDNmzDChferUqXLLLbfI/v37pVChQq7HacieNGmSCfPBwcHy7rvvyoEDB2Tx4sXyySefSFBQkKSkpLgC97p168wHA4MHDzZt0OfNSEJCgtmc4uLizNfQQIcEBf3vgwTkfVpP96+wbuSJt9rgC21BzqK29kVt7Yva2he19d/aJmaz5gEOZ/csMrWQ2rFjx+Snn34yvdBq1KhR8tlnn8nu3btNT7cutJZ6sbWDBw9Ks2bNzLEjR45M99wanIsWLSrz58+Xbt26mX36HHquadOmeYTw559/Xn7//XcznF2tWrVKunTpIocOHZIKFSqYfdqeOnXqmA8HmjZtmu5z6rnGjx+fZr+2wcph8gAAAADg6+Lj46VXr15y5swZM5o4s+jpzqLrr7/eFbhVixYtTC91cnJyusdrgTREd+3a1SNw69DvMWPGmJ5oDfL6eC2qDhN316RJkzTnrFSpkitwqz179piw7Qzcqnbt2ibE630Zhe6IiAgZPny4R0+3nmPi9kBJCgm64vcEvk17uCc0SZGxWwMlIYXrdOeUXZGdvN0E8+mrfuimU1VCQkK83RzkIGprX9TWvqitfVFb/61t3L8jgbOK0J0LNEjrEG/9VOTNN9/0uE+Hlp88eVJefvllE6JDQ0NNgE+9MFp6q6Hn1Arp+py6pabBLCmZcGY3WtcE6ppjfOmPrrbFl9qDnENt7Yva2he1tS9q63+1DclmvQndWRQTE+Nxe9OmTVK9enUztzq1YcOGyY8//mjmfjsXPHPasGGDzJw508zjVkeOHJETJ05kqU21atUyj9fNfXj56dOnTY83AAAAACB3sXp5Funwbx2SvXfvXlmwYIFZCG3IkCFpjps9e7YJ1bNmzTLD0Y8ePWq2c+fOmfs1qL///vtm+LcG+fvuu08KFCiQpTa1b99e6tWrZ86hK6DrPO4HH3zQrIie3vB0AAAAAIC1CN1ZpGH2woULZmE0XSFcA7euYJ6ariKuw8v1cl9ly5Z1bVOmTDH3v/POO3Lq1Clp3LixWdX8iSeekKuuuipLbdJQv3TpUilWrJi0bt3ahPBrrrlGPvzww2y/XgAAAABA5jG8PAvcL7+l189OLTY21vW9Xqdbt4w0atTIXKvb3Z133ulxO70F5nXFcee1vt3pZcs0eOeEmIh2UqJEiRw5F3xjgYjly5ebhb+YhwQAAADkDnq6AQAAAACwCKEbAAAAAACLELoBAAAAALAIoRsAAAAAAIsQugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLBFt1YuR9zaPWSFJwuLebgRwSGuSQyc1E6kaulITkAG83xxZiJ3X1dhMAAADg4+jpBgAAAADAIoTuLGrbtq0MHTrU280AAAAAAPgwQjcAAAAAABYhdAMAAAAAYBEWUrsC58+fl0GDBsknn3wihQoVkhEjRnjc//7778vLL78se/fulfDwcLn55ptl+vTpctVVV5n7o6Oj5aabbpLVq1fLU089Jbt375aGDRvK7NmzpWbNmuaYgwcPyvDhw2XTpk3m+WrVqiVRUVHSvn171/NUrlxZBgwYIAcOHJCPP/5YihUrJmPGjDH7VGxsrFSpUkUWL14sM2bMkJiYGKlevbrMmjVLWrRokeHrS0hIMJtTXFyc+Roa6JCgIEcOv5vwFq2n+1dkX2JiovhSO3ylPcg51Na+qK19UVv7orb+W9vEbNY8wOFw8C/wy3j00Udl2bJl8u6775og/fTTT8u6devk4YcfNuFa95ctW9YE6GPHjpnwXLRoUVm+fLlH6G7evLm88MILUqpUKRk4cKAkJyfLhg0bzDE7d+40gbtly5YSGhoq7733nkyZMsUE+YoVK7pC99mzZ2XChAnSsWNHWbRokYwePdqEeH1uZ+i+9tprzWM1cOv9W7ZsMUE9ODj9z1giIyNl/PjxafbPnz9fwsLCLH1vAQAAAMCXxcfHS69eveTMmTNSuHDhTD+e0H0Z586dkxIlSsgHH3wgd911l9n3999/y9VXX216mDV0p7Z161Zp2rSpCcgFCxb06Olu166dOUYDedeuXeXChQuSP3/+dJ+7bt26Jpw/9thjrtDdqlUr07OutHRlypQxgVmPc4but99+W/r27WuO0UBep04d2bNnjwnjV9rTXaFCBak9cqEkhXDJMLvQHu4JTVJk7NZASUjhkmE5YVdkJ/EF+unrqlWrpEOHDhISEuLt5iAHUVv7orb2RW3ti9r6b23j4uKkZMmSWQ7dDC+/DB32ffHiRdNL7VS8eHHXsHC1bds201usvdWnTp2SlJQUs//w4cNSu3Zt13H169d3fa8940p7xrUnW8O9nkN71P/8809JSkoygVzP4c79HAEBASZ06zkyOsb9eTIK3dqzrltqGsySuJ6z7WhduU53zvC1P7jaHl9rE3IGtbUvamtf1Na+qK3/1TYkm/VmIbVs0vnXnTp1Mp94zJs3zwzl/vTTT819GtYzKpYGZuUM6DpPXB/3/PPPy7fffis7duyQevXqXfIczvM4z3ElzwMAAAAAyD30dF9G1apVTYjVRcmcc6u1N3vfvn3Spk0b+fnnn+XkyZMyadIkMyTbObw8s3Rud+/eveW2224zt7XnW4eLAwAAAADyLnq6L0PnZOv86JEjR8rXX38tu3btMuE4MPB/b50G8Xz58pnVwn/55Rf57LPPzEJnmaWLnunq6NrDrcPUdaI+vdMAAAAAkLcRuq/Aiy++aBYw6969u7mE14033ijXXXeduU9XIp8zZ465hJfO39Yeb105PLNeeuklcwmwG264wTyPDllv3LixBa8GAAAAAJBbGF5+hb3dumK4c9VwpT3fTvfee6/Z3LkvCt+2bVuP20qv0+2+T1cm1550d4MHD/a4nd5wc+0Zdz9H6ufRS5dldYH6mIh2ZuV22GdVRl01X1fcZvEPAAAAIHfQ0w0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEW4Tjcy1DxqjSQFh3u7GcghoUEOmdxMpG7kSklIDvB2c/Ks2Eldvd0EAAAA5CH0dAMAAAAAYBFCt5u2bdvK0KFDbfdcAAAAAADvIHQDAAAAAGARQreFHA6HJCUlebsZAAAAAAAvIXSnoiH5sccekyJFikjJkiVl7NixJjyr999/X5o0aSKFChWSMmXKSK9eveTYsWOux0ZHR0tAQIB8+eWXct1110loaKisX79ezp8/Lw8++KAULFhQypYtK1OnTk3zvPq4JUuWeOwrWrSozJkzx3V748aN0rBhQ8mfP79phx6vj9uxY4frmHXr1kmzZs3Mc+tzjRo1iuAPAAAAAF7C6uWpzJ07V/r27SubN2+WrVu3yoABA6RixYrSv39/SUxMlAkTJkjNmjVN2B4+fLj07t1bli9f7nEODbpTpkyRa665RooVKyYjR440YXjp0qVy1VVXydNPPy3ff/+9CdBXKi4uTrp37y633HKLzJ8/X3799dc0c8J///13c7+26b333pOff/7ZtFtDemRkZIbnTkhIMJv7c6nQQIcEBf3vAwfkfVpP96/IGv094Ktt8sW2IXuorX1RW/uitvZFbf23tonZrDmhO5UKFSrItGnTTA+yhusff/zR3Nbw+vDDD7uO00D9yiuvSNOmTeXcuXOmF9vp2WeflQ4dOpjv9b533nlHPvjgA2nXrp0r2F999dWZapcGbW3TW2+9ZUJ07dq1TcjWdjnNnDnTtP/VV181x1577bXyxx9/yFNPPSXjxo2TwMD0BzZERUXJ+PHj0+wf0yhFwsKSM9VO+L4JTVK83YQ8LfWHbL5k1apV3m4CLEJt7Yva2he1tS9q63+1jY+Pz9Z5Cd2pXH/99SawOrVo0cIMB09OTjbDuLXHeOfOnXLq1ClJSflfeDl8+LAJwU469Nvp4MGDcvHiRWnevLlrX/HixU2gz4y9e/dK/fr1TeB20mHk7vbs2WPa697+li1bmuD/22+/mR779ERERJhee/eebg3vE7cHSlJIUKbaCd+lPdwauMduDZSEFK7TnVW7IjuJr9FPX/WPhH7YFxIS4u3mIAdRW/uitvZFbe2L2vpvbeP+HQmcVYTuK/TPP/9Ip06dzDZv3jwpVaqUCdt6W0O1u/Dw8EyfX4Oyc+64U24NXdH537qlpsEsKZlwZjda1wTqmmW+/EdW2+bL7UPWUVv7orb2RW3ti9r6X21DsllvFlJLJSYmxuP2pk2bpHr16mZ+9MmTJ2XSpEnSqlUrM3TbfRG1jFStWtUUyf282ku+b98+j+M0xP/555+u2/v37/cYxuAc6u4+93rLli0e56hVq5Z89913HuF9w4YNZuG3zA5nBwAAAABkH6E7Fe291qHWOpx7wYIFMmPGDBkyZIgZmp0vXz5z+5dffpHPPvvMLKp2OTrXWxdm08XUvv76a9m1a5dZ6Cz1/Oqbb77ZzMXevn27WcBt4MCBHp+o6ErpOpxdF3bTYeQrV640i7Up53DyRx99VI4cOSKPP/64+ZBAF2575plnzOvJaD43AAAAAMA6DC9PRS/tdeHCBTNfOigoyARuDboabPXyXbryuC6g1rhxYxN6//Of/1z2nC+++KKZV62rj2uv83//+185c+aMxzE6b7xPnz6mF71cuXLy8ssvy7Zt21z3Fy5cWD7//HMZNGiQWfW8Xr16ZnE0DePOed7ly5c3izxpwG/QoIGZO66Bf8yYMRa8UwAAAACAywlwpJ5IjDxD55ZrUNcAX6BAgRw7ry4UoNcpP3HihJQoUSLHzgvv0jUC9EMZvawc85DshdraF7W1L2prX9TWvqit/9Y27t98pLlLO0Mzi57uPESvva2XKtMebV1BXS8Fdvfdd+do4AYAAAAA5BxCdx5y9OhRM6Rcv5YtW1buuusuee6557zdLAAAAABABgjdeciTTz5pNgAAAABA3sCS1gAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE1cuRoeZRayQpONzbzUAOCQ1yyORmInUjV0pCcoC3m+PTYid19XYTAAAAYBP0dAMAAAAAYBFCdy5wOBwyYMAAKV68uAQEBMiOHTu83SQAAAAAQC5geHkuWLFihcyZM0eio6PlmmuukZIlS3q7SQAAAACAXEDozgUHDx6UsmXLyg033GDZcyQmJkpISIhl5wcAAAAAZB6h22K9e/eWuXPnmu91aHmlSpXkl19+kRdeeEHefPNNOXr0qNSoUUPGjh0rd955pxmKXr16dRk4cKCMGDHCdR4dkt6oUSPZv3+/VKtWzZxr5syZ8uWXX8qaNWtk5MiREhkZKUuXLpXx48fL7t27pVy5cvLQQw/J6NGjJTg441InJCSYzSkuLs58DQ10SFCQw9L3B7lH6+n+FZf+ECsvtjevtRuXR23ti9raF7W1L2rrv7VNzGbNAxya8mCZM2fOyCuvvGIC9pYtWyQoKMh8/8EHH8j06dNNwP7mm29MyF65cqW0adNGnn/+eZk3b5789NNPrvMMGTLEBO9169aZ2xq6r7rqKpk0aZJ5jIbqX3/9Vbp162aer1WrVqaHXeeSa/B/5plnMmyjhnUN6qnNnz9fwsLCLHpnAAAAAMD3xcfHS69evUy2K1y4cKYfT+jOBRqudYuNjTU9yrqg2urVq6VFixauY/r162eKqUH3jz/+kIoVK8rGjRulWbNm5pMV7bWeMmWK6bl2hu6hQ4fKtGnTXOdo3769tGvXTiIiIlz7NNw/+eST5pyZ6emuUKGC1B65UJJCuGSYXWgP94QmKTJ2a6AkpHDJsEvZFdlJ8hL9HbFq1Srp0KED00xshtraF7W1L2prX9TWf2sbFxdn1uXKauhmeHkuO3DggAnXWlB3Fy9eNMPHlQbsrl27yrvvvmtC9+eff25C8V133eXxmCZNmnjc3rlzp2zYsEGee+45177k5GT5559/zHNm1GsdGhpqttQ0mCVxPWfb0bpyne5Ly6t/SLXdebXtuDRqa1/U1r6orX1RW/+rbUg2603ozmXnzp0zX5ctWybly5f3uM89+GrP9wMPPGB6smfPni09e/ZME5rDw8PTnFuHid9+++1pnjd//vw5/EoAAAAAAJdD6M5ltWvXNuH68OHDZi52Rm655RYTql9//XVzyTGd9305jRs3lr1795qF1gAAAAAA3kfozmWFChUyq5IPGzZMUlJS5MYbbzRzA3RYuM4PcM7Z1gXXdAE0nZ+ti625z//OyLhx48xCajofXFdCDwwMNEPOd+3aJRMnTsyFVwcAAAAAcBfocQu5YsKECeYSYVFRUVKrVi3p3LmzGW5epUoVj+P69u1r5nr36dPnis7bqVMn+eKLL+Srr76Spk2byvXXX2+Gp+tlygAAAAAAuY+e7lygq4zr5qQrj+slwHS7lN9//91M2n/wwQfT3JfRovMavHXLCTER7aREiRI5ci74xqqMy5cvNytzs/gHAAAAkDsI3T5IVyo/fvy4uX62rlheunRpbzcJAAAAAJAFDC/3QQsWLDBDwk+fPi2TJ0/2dnMAAAAAAFlE6PZBuoCaXl9727ZtaS4rBgAAAADIOwjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEW4ZBgy1DxqjSQFh3u7GcghoUEOmdxMpG7kSklIDvB2c3xW7KSu3m4CAAAAbISebgAAAAAALELotljbtm1l6NChXnv+yMhIadiwodeeHwAAAAD8GaEbAAAAAACLELoBAAAAALAIoTsXpKSkyJNPPinFixeXMmXKmCHfTi+99JLUq1dPwsPDpUKFCvLoo4/KuXPnPIanBwQEpNliY2PN/adPn5Z+/fpJqVKlpHDhwnLzzTfLzp07vfI6AQAAAACeWL08F8ydO1eGDx8uMTEx8t1330nv3r2lZcuW0qFDBwkMDJRXXnlFqlSpIr/88osJ3RrQZ86caR77ySefyMWLF13nGjx4sPz0009SunRpc/uuu+6SAgUKyJdffilFihSRN954Q9q1ayf79u0zIf9KJCQkmM0pLi7OfA0NdEhQkCOH3w14i9bT/SvSl5iYKHm1zXmx7bg0amtf1Na+qK19UVv/rW1iNmse4HA4+Be4hbSnOjk5Wb799lvXvmbNmpke6UmTJqU5ftGiRTJw4EA5ceJEmvumTZsmzz77rAnvNWrUkPXr10vXrl3l2LFjEhoa6jquWrVqJrgPGDDA9KovWbJEduzYkWEb9Zjx48en2T9//nwJCwvL4isHAAAAgLwvPj5eevXqJWfOnDGjizOLnu5cUL9+fY/bZcuWNUFZrV69WqKiouTnn382PcxJSUnyzz//mMK6B17tyR41apR8/vnnJnArHUauQ9FLlCjhcf4LFy7IwYMHr7h9ERERpifeSduhQ90nbg+UpJCgLL9u+Bbt4Z7QJEXGbg2UhBSu052RXZGdJK/RT19XrVplRs+EhIR4uznIQdTWvqitfVFb+6K2/lvbuH9HAmcVoTsXpC6czsnWed46L7tbt24yaNAgee6558xwcO297tu3rxlS7gzdu3fvlnvuucf0jHfs2NF1Hg3cGuCjo6PTPGfRokWvuH3aS+7eU+6kwSwpmXBmN1rXBOqaobz8R1Tbnpfbj4xRW/uitvZFbe2L2vpfbUOyWW9Ctxdt27bNhO+pU6eaud3qo48+8jhGh5l3795d7rjjDhk2bJjHfY0bN5ajR49KcHCwVK5cOVfbDgAAAAC4PFYv9yKde61DGWbMmGEWUXv//fdl1qxZHsdo2NYeb513rQHbuek88fbt20uLFi2kR48e8tVXX5me840bN8ro0aNl69atXntdAAAAAID/IXR7UYMGDcwlw1544QWpW7euzJs3z8zvdvfNN9/Irl27pFKlSmYouXM7cuSIGaa+fPlyad26tfTp08fM9dZh6L/++qtrdXMAAAAAgPcwvNxi6c231tXEnXTIeOph4w888IDr+8stLl+oUCFzyTHd0qM95O7XBc+MmIh2aRZpQ96loyr0QxpdKIx5SAAAAEDuoKcbAAAAAACLELoBAAAAALAIoRsAAAAAAIsQugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLELoBAAAAALBIsFUnRt7XPGqNJAWHe7sZyCGhQQ6Z3EykbuRKSUgOEH8VO6mrt5sAAAAAP0JPNwAAAAAAFiF0W6ht27YydOhQ833lypVl+vTprvsCAgJkyZIl5vvY2Fhze8eOHTneBivPDQAAAAC4NIaX55ItW7ZIeDhDtQEAAADAnxC6c0mpUqUsPf/FixclX758lj4HAAAAACBzCN25RIeX61Bz53DzS0lOTpYBAwbI119/LUePHpWKFSvKo48+KkOGDHEd07t3bzl9+rQ0bdpUXnvtNQkNDZVDhw7J5s2b5ZFHHpE9e/ZI3bp1ZfTo0Zd9voSEBLM5xcXFma+hgQ4JCnJk+TXDt2g93b/6q8TERLHra7Lja/N31Na+qK19UVv7orb+W9vEbNac0O2DUlJS5Oqrr5aPP/5YSpQoIRs3bjQhvGzZsnL33Xe7jluzZo0ULlxYVq1aZW6fO3dOunXrJh06dJAPPvjAhHD3oJ6RqKgoGT9+fJr9YxqlSFhYcg6/OnjbhCYp4s+WL18uduX8XQD7obb2RW3ti9raF7X1v9rGx8dn67yEbh8UEhLiEYKrVKki3333nXz00UceoVvniL/99tuuYeVvvvmmCezvvPOO5M+fX+rUqSO//fabDBo06JLPFxERIcOHD/fo6a5QoYJM3B4oSSFBlrxG5D7t4dbAPXZroCSk+O8lw3ZFdhK70U9f9Y+EfuCmvz9gH9TWvqitfVFb+6K2/lvbuH9HAmcVodtH6ZDxd999Vw4fPiwXLlwwc7YbNmzocUy9evU85nHrkPL69eubwO3UokWLyz6XDk3XLTUNZkl+fD1nu9K6+vN1uu38R1Jfm51fnz+jtvZFbe2L2toXtfW/2oZks95cMswHLVy4UEaMGCF9+/aVr776ylzuq0+fPiZ4u2M1dAAAAADwbfR0+6ANGzbIDTfcYBZPczp48OBlH1erVi15//335Z9//nH1dm/atMnStgIAAAAAMkZPtw+qXr26bN26VVauXCn79u2TsWPHmut8X06vXr0kICBA+vfvL7t37zYLRk2ZMiVX2gwAAAAASIvQ7YP0kl+333679OzZU5o3by4nT5706PXOSMGCBeXzzz+XH3/8URo1amQuF/bCCy/kSpsBAAAAAGkxvNxC0dHRru9jY2M97nM4HB7X8Ha/rYuazZ4922ypL+3lNGfOnHSf8/rrrzdzwDN6rsyIiWhnLlkG+6zKqKMfdPVuFv8AAAAAcgc93QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITrdCNDzaPWSFJwuLebgRwSGuSQyc1E6kaulITkAPFHsZO6ersJAAAA8DP0dAMAAAAAYBFCt4+KjY2VgIAA2bFjxxU/JjIyUho2bGhpuwAAAAAAV47QbWO9e/eWHj16eLsZAAAAAOC3CN0AAAAAAFiE0J1LVqxYITfeeKMULVpUSpQoId26dZODBw+67t+8ebM0atRI8ufPL02aNJHt27d7PH7OnDnmse6WLFlihqBnNNR87ty5snTpUnOMbtHR0Ra9OgAAAABAeli9PJecP39ehg8fLvXr15dz587JuHHj5LbbbjNztuPj400I79Chg3zwwQdy6NAhGTJkSLaeb8SIEbJnzx6Ji4uT2bNnm33FixdP99iEhASzOeljVGigQ4KCHNlqB3yH1tP9qz9KTEwUO78uu74+f0Zt7Yva2he1tS9q67+1TcxmzQndueSOO+7wuP3uu+9KqVKlZPfu3bJx40ZJSUmRd955x/R016lTR3777TcZNGhQlp+vYMGCUqBAAROmy5Qpc8ljo6KiZPz48Wn2j2mUImFhyVluA3zThCYp4q+WL18udrZq1SpvNwEWobb2RW3ti9raF7X1v9rGx8dn67yE7lyyf/9+07sdExMjJ06cMCFbHT582PRIaw+4Bm6nFi1a5FrbIiIiTC+8e093hQoVZOL2QEkKCcq1dsBa2sOtgXvs1kBJSPHP63TviuwkdqSfvuofCR0tExIS4u3mIAdRW/uitvZFbe2L2vpvbeP+HQmcVYTuXNK9e3epVKmSvPXWW1KuXDkTuuvWrSsXL168oscHBgaKw+E5LDinhraEhoaaLTUNZknJ/hnO7EzrmuCndbX7H0h9fXZ/jf6K2toXtbUvamtf1Nb/ahuSzXqzkFouOHnypOzdu1fGjBkj7dq1k1q1asmpU6dc9+vtH374Qf755x/Xvk2bNnmcQ4einz171swNd7rcNbzz5csnyckMDwcAAAAAbyF054JixYqZFcvffPNNOXDggHz99dcew7l79eplVhfv37+/meOt806nTJnicY7mzZtLWFiYPP3002bV8/nz55sVzS+lcuXKJsxr4Nch7Sz6AAAAAAC5i9CdC3Ro+MKFC2Xbtm1mSPmwYcPkxRdf9Fj07PPPP5cff/zRXDZs9OjR8sILL3icQ1ce15XNNZDXq1dPFixYYC4Ldika4mvWrGkuQaY95Rs2bLDsNQIAAAAA0mJOdy5p37696cV25z5H+/rrr08zXDz1HO4ePXqYLXWwdtIQ7h7ENWh/9dVXWW5zTEQ700MPe9CRDvqhjS4mxjwkAAAAIHfQ0w0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWCTYqhMj72setUaSgsO93QzkkNAgh0xuJlI3cqUkJAeIv4md1NXbTQAAAIAfoqcbAAAAAACLELoBAAAAALAIoRsAAAAAAIsQugEAAAAAsAgLqVlg0aJFMn78eDlw4ICEhYVJo0aNZOnSpRIeHi7vvvuuTJ061dxXvHhxueOOO+TVV181jzt8+LA8/vjjsmbNGgkMDJTOnTvLjBkzpHTp0q5zv/766zJlyhQ5cuSIVKlSRcaMGSMPPPCA6/6AgACZOXOmfPbZZxIdHS1ly5aVyZMny5133plhexMSEszmFBcXZ76GBjokKMhh0buE3Kb1dP/qbxITE8Xur83Or9FfUVv7orb2RW3ti9r6b20Ts1nzAIfD4Z//ArfIn3/+KRUrVjRB97bbbpOzZ8/Kt99+Kw8++KC8//77Mnz4cJk0aZJ06dJFzpw5Ixs2bJChQ4dKSkqKXHfddVKwYEGZPn26JCUlyeDBg81tDc/q008/lZ49e5r727dvL1988YU8+eSTsmrVKrnppptcobtEiRLmOVq3bm2eMyoqSn788UepVatWum2OjIw0HxKkNn/+fPOhAQAAAAD4q/j4eOnVq5fJb4ULF8704wndOez777834Tk2NlYqVarkcV/58uWlT58+MnHixDSP0+CsQfzQoUNSoUIFs2/37t1Sp04d2bx5szRt2lRatmxpbr/55puux919991y/vx5WbZsmSt0Dxw40PSIO11//fXSuHFj0wN+pT3d2obaIxdKUgiXDLML7eGe0CRFxm4NlIQU/7tk2K7ITmJX+umr/g7p0KGDhISEeLs5yEHU1r6orX1RW/uitv5b27i4OClZsmSWQzfDy3NYgwYNpF27dlKvXj3p1KmTdOzY0Qzt1kL+8ccf5r707NmzxwRdZ+BWtWvXlqJFi5r7NHTr1wEDBng8ToP4yy+/7LGvRYsWaW7v2LEjwzaHhoaaLTUNZkl+eD1nu9O6+uN1uv3hj6O+Rn94nf6I2toXtbUvamtf1Nb/ahuSzXqzkFoOCwoKMp+SfPnllyY065zsmjVryl9//eXtpgEAAAAAchmh2wI6xFt7oHWe9Pbt2yVfvnwmiFeuXNkskpYenW+ti6Pp5qTDy0+fPm3Cu/MYnQPuTm8773fatGlTmtsZzecGAAAAAFiH4eU5LCYmxgRrHVZ+1VVXmdvHjx83oVcXLNP51rpf52/rImsamnXFcl0YTYek33fffa6F1B599FFp06aNNGnSxJx75MiRZg63roaux3/++efyySefyOrVqz3a8PHHH5vH3HjjjTJv3jwzJ/ydd97x0jsCAAAAAP6L0J3DdGL9N998Y4KzTrjXxdT0EmEastU///wj06ZNkxEjRpjJ+M5LeWnvuF5WTAO4rjrufskwpx49epj523rJsCFDhphLhs2ePVvatm3r0QbtYV+4cKEJ7XrJsAULFqTpDQcAAAAAWI/QncO0R3vFihUZ3v/II4+YLT16qTEN3pcyaNAgs11KuXLl5KuvvpLsioloZy4/BnvQxfyWL19uVvFm8Q8AAAAgdzCnGwAAAAAAixC6AQAAAACwCMPLbcbhcHi7CQAAAACAf9HTDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiEhdSQoeZRayQpONzbzUAOCQ1yyORmInUjV0pCcoDYXeykrt5uAgAAAEBPNwAAAAAAViF0AwAAAABgEUI3AAAAAAAWIXQDAAAAAGARQncekZCQIE888YRcddVVkj9/frnxxhtly5Yt5r4mTZrIlClTXMf26NFDQkJC5Ny5c+b2b7/9JgEBAXLgwAGvtR8AAAAA/BGrl+cRTz75pCxevFjmzp0rlSpVksmTJ0unTp1MkG7Tpo1ER0fLiBEjxOFwyLfffitFixaV9evXS+fOnWXdunVSvnx5qVatWoaBXjenuLg48zU00CFBQY5ce42wltbT/avdJSYmir9wvlZ/es3+gtraF7W1L2prX9TWf2ubmM2aBzg0pcGnnT9/XooVKyZz5syRXr16uQpfuXJlGTp0qFx77bXywAMPyMmTJ2XXrl0maPfs2dP0iE+aNEn69+8v8fHxMm/evHTPHxkZKePHj0+zf/78+RIWFmb56wMAAAAAX6VZSnPYmTNnpHDhwpl+PD3decDBgwdNyG7ZsqVrnw4fb9asmezZs8eE6rNnz8r27dtl48aNpue7bdu2JnAr7ekeOXJkhuePiIiQ4cOHe/R0V6hQQSZuD5SkkCCLXx1yi/ZwT2iSImO3BkpCiv2v070rspP4C/39sGrVKunQoYP53QD7oLb2RW3ti9raF7X139rG/TsSOKsI3TagQ8kbNGhghph/99135n+W1q1bm97uffv2yf79+00Qz0hoaKjZUtNglpRs/3Dmb7SuCX5QV3/8Y6iv2R9ftz+gtvZFbe2L2toXtfW/2oZks94spJYHVK1aVfLlyycbNmzw+DRGF1KrXbu2ua2heu3atfLNN9+YXu7ixYtLrVq15LnnnpOyZctKjRo1vPgKAAAAAMA/EbrzgPDwcBk0aJAZIr5ixQrZvXu3a5523759zTEatFeuXCnBwcFmjrdzn87jvlQvNwAAAADAOoTuPELnZ99xxx1mwbTGjRubVcs1ZOsCa6pVq1aSkpLiEbA1dCcnJ5uvAAAAAIDcx5zuPEJXIn/llVfMlh4dTq6h251erzs7i9PHRLSTEiVKZPnx8C06JWH58uVmgTHmIQEAAAC5g55uAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALELoBgAAAADAIoRuAAAAAAAsQugGAAAAAMAiwVadGHlf86g1khQc7u1mIIeEBjlkcjORupErJSE5QOwodlJXbzcBAAAA8EBPNwAAAAAAFiF0AwAAAABgEUI3AAAAAAAWIXTnARcvXvR2EwAAAAAAWcBCatnUtm1bqVevngQFBcncuXMlX758MnHiROnVq5c89thjsmjRIildurTMmDFDunTpYh6zbt06GTlypOzcuVOKFy8uDz30kHlMcHCw65x169Y1tz/44ANz/rVr18pLL70ks2fPll9++cU8rnv37jJ58mQpWLCgedycOXNk6NCh8uGHH5qvR44ckRtvvNE8pmzZshm+hoSEBLM5xcXFma+hgQ4JCnJY/A4it2g93b/aUWJiovgj5+v219dvZ9TWvqitfVFb+6K2/lvbxGzWPMDhcNj3X+C5QAPy999/L08++aT07NnTBN7IyEjp2LGj3Hbbbeb+adOmyUcffSSHDx+WU6dOSY0aNaR3797y+OOPy88//yz9+/eXwYMHm8c5z7lt2zYZNGiQ9O3b1+yrWbOmTJ8+XRo0aCBVqlQxwfvRRx+Vm2++WWbOnOkK3QMGDJA2bdpIVFSUBAYGyv333y+NGjWSefPmZfga9HnHjx+fZv/8+fMlLCzMsvcOAAAAAHxdfHy86VQ9c+aMFC5cONOPJ3Rnkwbk5ORk+fbbb81t/b5IkSJy++23y3vvvWf2HT161PQ0f/fdd/L555/L4sWLZc+ePRIQ8L/LNmlofuqpp0wRNSjrObW3WcP8pWgv+sCBA+XEiROu0N2nTx85cOCAVK1a1XXuZ5991rQhMz3dFSpUkNojF0pSCJcMswvt4Z7QJEXGbg2UhBR7XjJsV2Qn8Uf66euqVaukQ4cOEhIS4u3mIAdRW/uitvZFbe2L2vpvbePi4qRkyZJZDt0ML88B9evXd32vw8xLlChhhoQ76fBydezYMRO2W7Ro4QrcqmXLlnLu3Dn57bffpGLFimbfddddl+Z5Vq9ebXqwtXdcC5+UlCT//POP+eTF2SOtX52BW2nY1+e9lNDQULOlpsEsyabXc/ZnWle7Xqfb3/8A6uv39/fArqitfVFb+6K29kVt/a+2IdmsNwup5YDURdBA7b7PGbBTUlKu+Jzh4Z49zLGxsdKtWzcT8LWnXIefv/baa2kWWkuvLQxmAAAAAADvoKc7l9WqVcuEZg3CzjC+YcMGKVSokFx99dUZPk5Dtob2qVOnmiHoSueJAwAAAAB8Fz3duUwXP9NVxZ2LqC1dulSeeeYZGT58uCtMp6datWpmroGugq6LqL3//vsya9asXG07AAAAACBzCN25rHz58rJ8+XLZvHmzWYlcF0LTFcrHjBlzycfpsXrJsBdeeMFcTkxXI9f53QAAAAAA38Xw8myKjo5Os0/nX6fmPq9aL+mloTsz51TDhg0zm7sHHnjA9b1ehkw3dz169MjynO6YiHZmUTjYg46U0A98dIVvFv8AAAAAcgc93QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARYKtOjHyvuZRayQpONzbzUAOCQ1yyORmInUjV0pCcoDYTeykrt5uAgAAAJAGPd0AAAAAAFiE0O0lbdu2laFDh3q7GQAAAAAACxG6AQAAAACwCKHbC3r37i3r1q2Tl19+WQICAswWGxsru3btki5dukjBggWldOnS8sADD8iJEydcj0tJSZGoqCipUqWKFChQQBo0aCCLFi1y3R8dHW3OtWbNGmnSpImEhYXJDTfcIHv37vXSKwUAAAAA/8ZCal6gYXvfvn1St25defbZZ82+kJAQadasmfTr10+mTZsmFy5ckKeeekruvvtu+frrr80xGrg/+OADmTVrllSvXl2++eYbuf/++6VUqVLSpk0b1/lHjx4tU6dONfsHDhwoDz/8sGzYsCHD9iQkJJjNKS4uznwNDXRIUJDDwncCuUnr6f7VbhITE8VfOV+7P78HdkVt7Yva2he1tS9q67+1TcxmzQMcDoc9/wWeB+Z0N2zYUKZPn25uT5w4Ub799ltZuXKl65jffvtNKlSoYHqqK1WqJMWLF5fVq1dLixYtXMdoSI+Pj5f58+ebnu6bbrrJHNOuXTtz//Lly6Vr164mxOfPnz/dtkRGRsr48ePT7Ndzam85AAAAAPir+Ph46dWrl5w5c0YKFy6c6cfT0+0jdu7cKWvXrjVDy1M7ePCg+XRFi92hQweP+y5evCiNGjXy2Fe/fn3X92XLljVfjx07JhUrVkz3uSMiImT48OEePd0a9iduD5SkkKBsvzb4Bu3hntAkRcZuDZSEFPtdMmxXZCfxV/r7YdWqVeb3g46agX1QW/uitvZFbe2L2vpvbeP+HQmcVYRuH3Hu3Dnp3r27vPDCC2nu0+Cs873VsmXLpHz58h73h4aGetx2/x9F53g754NnRB+f+hxKg1mSDa/n7O+0rna8Tjd//P73HvA+2BO1tS9qa1/U1r6orf/VNiSb9SZ0e0m+fPkkOTnZdbtx48ayePFiqVy5sgQHpy1L7dq1TTA+fPiwx/xtAAAAAIDvYvVyL9FwHRMTY1Yt1xXKBw8eLH///bfce++9smXLFjOkXOd39+nTx4TzQoUKyYgRI2TYsGEyd+5cc//3338vM2bMMLcBAAAAAL6H0O0lGqCDgoJMD7auMq5zs3WFcQ3YHTt2lHr16snQoUOlaNGiEhj4vzJNmDBBxo4da1Yxr1WrlnTu3NkMN9dLiAEAAAAAfA/Dy72kRo0a8t1336XZ/8knn2T4GJ2fPWTIELNltCJ66sXodYX0rC5QHxPRTkqUKJGlx8I3F4jQ1ex1wTHmIQEAAAC5g55uAAAAAAAsQugGAAAAAMAihG4AAAAAACxC6AYAAAAAwCKEbgAAAAAALELoBgAAAADAIoRuAAAAAAAsQugGAAAAAMAiwVadGHlf86g1khQc7u1mIIeEBjlkcjORupErJSE5QOwkdlJXbzcBAAAASBc93QAAAAAAWITQ7UWVK1eW6dOne7sZAAAAAACLELqzoW3btjJ06NAsP37Lli0yYMAA1+2AgABZsmSJxzGRkZHSsGHDbLUTAAAAAOAdzOn2olKlSuXac128eFHy5cuXa88HAAAAAKCnO8t69+4t69atk5dfftn0UOtWsmRJmTJliuuYHj16SEhIiGW/+0QAAC9MSURBVJw7d87c/u2338xxBw4cSDO8XL9Xt912mzlGb8+ZM0fGjx8vO3fudD2H7lOnT5+Wfv36meBeuHBhufnmm81xqXvI3377balSpYrkz58/V98fAAAAAAA93VmmYXvfvn1St25defbZZ82+F154QaKjo2XEiBHicDjk22+/laJFi8r69eulc+fOJqSXL19eqlWrlu5Q86uuukpmz55tjg0KCpKCBQvKrl27ZMWKFbJ69WpzXJEiRczXu+66SwoUKCBffvml2ffGG29Iu3btTJuKFy9ujtFwv3jxYvnkk0/M+TKSkJBgNqe4uDjzNTTQIUFBjhx+5+AtWk/3r3aSmJgo/sz5+v39fbAjamtf1Na+qK19UVv/rW1iNmtO6M4iDbo6XDssLEzKlClj9mlvs4bm5ORkE5b1/p49e5ogrkFav7Zp0+aSQ801pDvPpzR4BwcHe+zTEL9582Y5duyYhIaGmn3aw67zwRctWuSaJ65Dyt97773LDmOPiooyPeqpjWmUImFhyVl6f+C7JjRJEbtZvny5t5vgE1atWuXtJsAi1Na+qK19UVv7orb+V9v4+PhsnZfQnYNatWolZ8+ele3bt8vGjRtNwNbF1iZNmmTu157ukSNHZvt5dBi5DlkvUaKEx/4LFy7IwYMHXbcrVap0RfPGIyIiZPjw4R493RUqVJCJ2wMlKSTjHnLkLdrDrYF77NZASUix13W6d0V2En+mn77qH4kOHTqYKS2wD2prX9TWvqitfVFb/61t3L8jgbOK0J2DtJe6QYMGpkf7u+++M0Vr3bq16e3WYd/79+/PsKc7MzRwly1b1jxPem1wCg8Pv6LzaW+5s8fcnQazpGR7hTP8r64JNqsrf/j+/33gvbAnamtf1Na+qK19UVv/q21INutN6M4GHT6uQ8ndaaheu3atGf793HPPmfnVtWrVMt9rUK5Ro0aG59Nipj5fes/RuHFjOXr0qBl27lyADQAAAADge1i9PBs08MbExEhsbKycOHFCUlJSzHDylStXmkB87bXXmuN037x58y7by63nW7NmjQnUp06dcu07dOiQ7NixwzyHLnjWvn17adGihVkd/auvvjLPr8PZR48eLVu3bs2V1w4AAAAAuDxCdzboKuW6Knjt2rXN3OnDhw+bed0avt0DtoZu7a3Wr5cydepUM5dA51M3atTI7LvjjjvMImw33XSTeY4FCxaYS4fpwlE6dL1Pnz6m9/yee+6RX3/9VUqXLm356wYAAAAAXBmGl2eDhl2du52ahm532iOtlxBLTXuo3XXv3t1s7nSuta5InlqhQoXklVdeMVt69DrdumVHTES7NIu1IW8vEKEf1uiiY8xDAgAAAHIHPd0AAAAAAFiE0A0AAAAAgEUI3QAAAAAAWITQDQAAAACARQjdAAAAAABYhNANAAAAAIBFCN0AAAAAAFiE0A0AAAAAgEWCrTox8r7mUWskKTjc281ADgkNcsjkZiJ1I1dKQnKA2EHspK7ebgIAAABwSfR0AwAAAABgEUK3j5gzZ44ULVrU280AAAAAAOQgQreP6Nmzp+zbt8/bzQAAAAAA5CDmdPuAxMREKVCggNkAAAAAAPZB6LZISkqKTJkyRd588005cuSIlC5dWh555BG57777pEqVKrJw4UKZOXOmxMTEyKxZs8xjhg4dKqdPnzbfR0ZGypIlS+SJJ54w3//999/y4IMPyowZM2Tq1Kny0ksvmecYMmSIjB492vW8+vgRI0bI0qVLJSEhQZo0aSLTpk2TBg0aZNhWPU43p7i4OPM1NNAhQUEOC98l5Catp/tXu3xghf9/H3g/7Ifa2he1tS9qa1/U1n9rm5jNmhO6LRIRESFvvfWWCbw33nij/Pnnn/Lzzz+77h81apQJz40aNZL8+fPLypUr05zj4MGD8uWXX8qKFSvM93feeaf88ssvUqNGDVm3bp1s3LhRHn74YWnfvr00b97cPOauu+4yPeb6uCJFisgbb7wh7dq1M0PXixcvnm5bo6KiZPz48Wn2j2mUImFhyTn6vsD7JjRJEbtYvny5t5vgU1atWuXtJsAi1Na+qK19UVv7orb+V9v4+PhsnTfA4XDYp9vLR5w9e1ZKlSolr776qvTr18/jvtjYWNPTPX36dNNL7b6QWuqe7hdffFGOHj0qhQoVMvs6d+4se/fuNQE8MPB/0/GvvfZa6d27twnx69evl65du8qxY8ckNDTUde5q1arJk08+KQMGDLjinu4KFSpI7ZELJSmES4bZhfZwa+AeuzVQElLsccmwXZGdvN0En6CfvuofiQ4dOkhISIi3m4McRG3ti9raF7W1L2rrv7WNi4uTkiVLypkzZ6Rw4cKZPj893RbYs2ePCbHaw5wRHfZ9OZUrV3YFbqVD1IOCglyB27lPQ7bauXOnnDt3TkqUKOFxngsXLpignhEN6O4h3UmDWZJNrucMz7ra5Trd/MFL+37wntgTtbUvamtf1Na+qK3/1TYkm/UmdFvgShZECw+/fA9y6uIGBASku0/ndisN3GXLlpXo6Og05+JyZAAAAACQ+wjdFqhevboJ3mvWrEkzvNxKjRs3NsPRg4ODTS85AAAAAMC7CN0W0IXRnnrqKTOPOl++fNKyZUs5fvy4/PTTT5cccp5duqBaixYtpEePHjJ58mSz4Noff/why5Ytk9tuu+2KhrQDAAAAAHIOodsiY8eONT3O48aNM8FXh30PHDjQ0ufUoea6mrNeQqxPnz4m6JcpU0Zat25t5n5nVkxEuzTzw5G3F4jQ/z908THmIQEAAAC5g9BtEV3sTMOv+zW0ndJbMF5XINfNSVcv182drnCeWur527rw2iuvvGI2AAAAAIB3/f8y2AAAAAAAIEcRugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLELoBAAAAALAIoRsAAAAAAItwnW5kqHnUGkkKDvd2M5BDQoMcMrmZSN3IlZKQHCB5Reykrt5uAgAAAJBl9HQDAAAAAGARQrfNRUdHS0BAgJw+fdrbTQEAAAAAv0Po9kEEZQAAAACwB0I3AAAAAAAWIXRnUdu2beWxxx4zW5EiRaRkyZIyduxYcTgc5v5Tp07Jgw8+KMWKFZOwsDDp0qWL7N+/3/X4X3/9Vbp3727uDw8Plzp16sjy5cslNjZWbrrpJnOM3qc93r179za3U1JSJCoqSqpUqSIFChSQBg0ayKJFizzapeeoUaOGuV/Po+cDAAAAAHgHq5dnw9y5c6Vv376yefNm2bp1qwwYMEAqVqwo/fv3N0FZQ/Znn30mhQsXlqeeekpuueUW2b17t4SEhMjgwYPl4sWL8s0335jQrfsLFiwoFSpUkMWLF8sdd9whe/fuNY/VAK00cH/wwQcya9YsqV69unns/fffL6VKlZI2bdrIkSNH5Pbbbzfn1rZom/773/9e9nUkJCSYzSkuLs58DQ10SFDQ/z5EQN6n9XT/mlckJiZ6uwl55j3ivbIfamtf1Na+qK19UVv/rW1iNmse4HB2zSLTPd3Hjh2Tn376yfRGq1GjRpmQvXTpUtPbvGHDBrnhhhvMfSdPnjSBWoP6XXfdJfXr1zfB+plnnkl3Trf2UmtvedGiRc0+DcXFixeX1atXS4sWLVzH9uvXT+Lj42X+/Pny9NNPm+fWNjlpm1544QWPc6UWGRkp48ePT7Nfz6m99AAAAADgr+Lj46VXr15y5swZ0ymaWfR0Z8P111/vCtxKw/DUqVNNr3VwcLA0b97cdV+JEiWkZs2asmfPHnP7iSeekEGDBslXX30l7du3NwFcg3hGDhw4YIrdoUMHj/3aW96oUSPzvZ7b/TmdbbqciIgIGT58uEdPt35AMHF7oCSFBF3RewHfpz3cE5qkyNitgZKQkneu070rspO3m+Dz9NPXVatWmd8POpIG9kFt7Yva2he1tS9q67+1jft3JHBWEbq9RHuoO3XqJMuWLTPBW4eOa2B//PHH0z3+3Llz5qseX758eY/7QkNDs9UWfXx659BglpScd8IZrozWNSEP1ZU/apl7r3i/7Ina2he1tS9qa1/U1v9qG5LNerOQWjbExMR43N60aZOZa127dm1JSkryuF+Hl+scbb3PSXuTBw4cKJ988omZe/3WW2+Z/fny5TNfk5OTXcfq4zQYHz58WKpVq+ax6XlUrVq1zPzy1G0CAAAAAHgHoTsbNADrsGwN0wsWLJAZM2bIkCFDTPC+9dZbzYJq69evl507d5oFz7SHWveroUOHysqVK+XQoUPy/fffy9q1a01oVpUqVTLD1r/44gs5fvy46eUuVKiQjBgxQoYNG2bmhR88eNA8Tp9TbysN8Lp428iRI02bdE72nDlzvPoeAQAAAIA/I3Rng14S7MKFC9KsWTOzYrgGbl01XM2ePVuuu+466datm5lXrevV6eW8nEMTtBdbH6NBu3PnzmbhtZkzZ5r7NJzrwma6CFrp0qXNZcnUhAkTzGXJdCi683E63FwvIaZ05XRd+XzJkiXmcmK6yvnzzz/vtfcHAAAAAPwdc7qzQQP09OnT5fXXX09zn15j+7333svwsdpDfSkarnVzp73fGux1y4iGfN3c9enTR7IiJqKdWQAO9lkgQj/40YXJmIcEAAAA5A56ugEAAAAAsAihGwAAAAAAizC8PIuio6O93QQAAAAAgI+jpxsAAAAAAIsQugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLsHo5MtQ8ao0kBYd7uxnIIaFBDpncTKRu5EpJSA4QXxM7qau3mwAAAADkOHq6AQAAAACwCKE7A7GxsRIQECA7duy44sc4HA4ZMGCAFC9ePNOPBQAAAADYD6FbRHr37i09evTw2FehQgX5888/pW7duld8nhUrVsicOXPkiy++yPRjL6Vy5coyffr0HDkXAAAAACD3MKc7A0FBQVKmTJlMPebgwYNStmxZueGGG8QXXbx4UfLly+ftZgAAAACA3/Cr0L1o0SIZP368HDhwQMLCwqRRo0Zmmzt3rrlfh4SrtWvXmt7lKlWqyPbt26Vhw4Zm/65du2TkyJHy7bffSnh4uHTs2FGmTZsmJUuWNL3l7uepVKmSPP300xIZGSm//fabBAb+/6CCW2+9VUqUKCHvvvuuCerDhw+XTZs2yfnz56VWrVoSFRUl7du3N8e2bdtWfv31Vxk2bJjZnMPY1fr16yUiIkK2bt1q2nDbbbeZx2rblL6Gvn37yv79+2XJkiVy++23m5741BISEszmFBcXZ76GBjokKOh/z4W8T+vp/tXXJCYmersJeZbzveM9tB9qa1/U1r6orX1RW/+tbWI2ax7gcCY4m9Ph3hUrVpTJkyebcHr27FkTnh988EETTDVozp492xyrc7L/+OMPj9B9+vRpqVGjhvTr18885sKFC/LUU09JUlKSfP3113LmzBl55ZVX5M0335QtW7aYnvLg4GDTW758+XJp166dOffff/9tesOd+3bu3GkCd8uWLSU0NFTee+89mTJliuzdu9e0V49v0KCBmSvev39/cw49p4Z13T9x4kTp2rWrHD9+XB577DGzz/k6NHSfOnVKxo0b5xo+X7Vq1TTvjX4woB9GpDZ//nzz4QQAAAAA+Kv4+Hjp1auXyXyFCxfO9OP9JnR///33ct1115kF0rQX2p32Umuo1t5gJz3OPXRruNWQvnLlStcx2oOtc781IGsg13nXuuljnTTsaq/2O++8Y25rKNeAe+TIEY/eb3c6F3zgwIEmRDvD89ChQ83mpOFfg/0bb7zh2qc9323atDE95vnz5zeP0578Tz/99JLvTXo93fq6ao9cKEkhXDLMLrSHe0KTFBm7NVASUnzvkmG7Ijt5uwl5ln76umrVKunQoYOEhIR4uznIQdTWvqitfVFb+6K2/lvbuLg4M7I4q6Hbb4aXaw+w9izXq1dPOnXqZIaG33nnnVKsWLErerz2SOuw84IFC6a5T3udNXSn57777jM91DNnzjQ92fPmzZN77rnHFbjPnTtnepqXLVtmeuO151x70Q8fPnzZ9vzwww/mfE76+UlKSoocOnTIDFNXTZo0uexr03bplpoGsyQfvJ4zskfr6ovX6eaPV868h7yP9kRt7Yva2he1tS9q63+1Dclmvf0mdGuvsH56sXHjRvnqq69kxowZMnr0aImJibmix2s47t69u7zwwgtp7tPh4hnRx2gY1lDdtGlT01uu88CdRowYYdqlQ8qrVasmBQoUMB8G6KJnl2vPI488Ik888USa+3RYupNzfjcAAAAAIPf5Teh2LnCmc6d103nOOsxch17rit7JycmXfGzjxo1l8eLFZsi2ztW+UjrMWxcw0x5pXcCtZs2a5lxOGzZsMMPbdZ65M0y7D09X6bVPz7F7924T1AEAAAAAvslvrtOtPdrPP/+8Welbh25/8sknZvExHYatQVqHauvc7BMnTqS7Ot3gwYPNomb33nuvWShNh5Tr/O4+ffpcNrDrEHPt6dbVyvV7d9WrVzdt2bFjhxkyrhP0dYi4O23fN998I7///rtpn9JF3LTXXud962N1hfKlS5e65oEDAAAAALzPb0K3TnjX4HrLLbeY+ddjxoyRqVOnSpcuXcyca+2B1vnPpUqVMr3PqZUrV87s14Ct88F1brgubFa0aNEMF0Rzuvnmm82K6BrqNVS7e+mll8y8cr22tw5F1/nm7j3h6tlnnzW937ryuLZP1a9fX9atWyf79u2TVq1amQXTtPde2wkAAAAA8A1+s3o5rpyuzlekSBHTq64rr8MedASHXqpOP3hi8Q97obb2RW3ti9raF7W1L2rrv7WN+zcfZXX1cr/p6QYAAAAAILcRugEAAAAAsAihGwAAAAAAixC6AQAAAACwCKEbAAAAAACLELoBAAAAALAIoRsAAAAAAIsEW3Vi5H3No9ZIUnC4t5uBHBIa5JDJzUTqRq6UhOQA8RWxk7p6uwkAAACAZejpBgAAAADAIoRuL2jbtq0MHTrUdbty5coyffp0r7YJAAAAAJDzGF7uA7Zs2SLh4QzjBgAAAAC7IXT7gFKlSnm7CQAAAAAACzC8/AqGgj/++ONmOHixYsWkdOnS8tZbb8n58+elT58+UqhQIalWrZp8+eWXrsfs2rVLunTpIgULFjTHP/DAA3LixIkMn8N9eLnD4ZDIyEipWLGihIaGSrly5eSJJ55wHTtz5kypXr265M+f35z7zjvvdN23YsUKufHGG6Vo0aJSokQJ6datmxw8eNCy9wYAAAAAcGn0dF+BuXPnypNPPimbN2+WDz/8UAYNGiSffvqp3HbbbfL000/LtGnTTLA+fPiwXLx4UW6++Wbp16+f2X/hwgV56qmn5O6775avv/76ss+1ePFi87iFCxdKnTp15OjRo7Jz505z39atW00Af//99+WGG26Qv//+W7799lvXY/WDgOHDh0v9+vXl3LlzMm7cONPGHTt2SGBgxp+vJCQkmM0pLi7OfA0NdEhQkCOb7x58hdbT/auvSExM9HYT8jzne8h7aT/U1r6orX1RW/uitv5b28Rs1jzAoV2ruGRPd3Jysivc6vdFihSR22+/Xd577z2zT4Nx2bJl5bvvvpPVq1ebY1euXOk6x2+//SYVKlSQvXv3So0aNcw5GzZs6Ord1p5u7UnX7aWXXpI33njD9JaHhIR4tOWTTz4xvet6Pu1hvxztXdeh6z/++KPUrVs3w+O0Z338+PFp9s+fP1/CwsIy8W4BAAAAgL3Ex8dLr1695MyZM1K4cOFMP56e7iugPcdOQUFBZuh2vXr1XPt0mLc6duyY6ZVeu3atGVqemg711tB9KXfddZcJ49dcc4107txZbrnlFunevbsEBwdLhw4dpFKlSq77dNOebGcw3r9/v+ndjomJMYE7JSXF7Nce+EuF7oiICNND7t7TrR8STNweKEkhQZl6r+C7tId7QpMUGbs1UBJSfOc63bsiO3m7CXmefvq6atUq8zsi9Yd1yNuorX1RW/uitvZFbf23tnH/jgTOKkL3FUj9xgcEBHjs09tKQ64O69aQ/MILL6Q5j/aGX46zR1x7zLXwjz76qLz44ouybt0607v9/fffS3R0tHz11VcmYGsvta5+rvO49Xk1lOucc50Lru3RsK1D3i9F547rlpoGs6Rk3wlnyBla1wQfqit/tHL2veT9tCdqa1/U1r6orX1RW/+rbUg2603ozmGNGzc287J1yLj2TmdFgQIFTIDWbfDgwXLttdeaIeJ6bj1n+/btzfbMM8+YsK1zxdu0aWPCugbuVq1amfOsX78+h18dAAAAACAzCN05TEOyBt97773XLL5WvHhxOXDggFkY7e233zbD0y9lzpw5Zt548+bNzbDxDz74wIRw7cH+4osv5JdffpHWrVubldSXL19uerNr1qxpbuuw9zfffNP0qOuQ8lGjRuXa6wYAAAAA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",
      "text/plain": [
       "<Figure size 1000x1200 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot feature importance\n",
    "fig, ax = plt.subplots(figsize=(10, 12))\n",
    "xgb.plot_importance(\n",
    "    gs_bow_pipe.best_estimator_.named_steps['xgbregressor'],\n",
    "    max_num_features=50,\n",
    "    ax=ax,\n",
    "    importance_type='gain',\n",
    "    show_values=False,\n",
    "    height=0.6  )\n",
    "\n",
    "tick_labels = [item.get_text() for item in ax.get_yticklabels()]\n",
    "\n",
    "passthrough_columns = X_train.drop(columns=['tokenized_words']).columns.tolist()\n",
    "\n",
    "for i in range(len(tick_labels)):\n",
    "    try:\n",
    "        feature_index = int(tick_labels[i][1:])\n",
    "\n",
    "        if feature_index < num_bow_features:\n",
    "            tick_labels[i] = feature_names[feature_index]\n",
    "        else:\n",
    "            passthrough_index = feature_index - num_bow_features\n",
    "\n",
    "            if 0 <= passthrough_index < len(passthrough_columns):\n",
    "                tick_labels[i] = passthrough_columns[passthrough_index]\n",
    "\n",
    "    except ValueError:\n",
    "        pass\n",
    "\n",
    "# Set updated tick labels\n",
    "ax.set_yticklabels(tick_labels)\n",
    "\n",
    "plt.title(\"Feature Importance - XGBoost (BOW without undersampling)\", fontsize=16)\n",
    "plt.xlabel(\"F score\", fontsize=12)\n",
    "plt.ylabel(\"Features\", fontsize=12)\n",
    "plt.xticks(fontsize=10)\n",
    "plt.yticks(fontsize=10)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"rf_bow.png\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict\n",
    "predictions = gs_bow_pipe.predict(X_test)\n",
    "predictions = list(map(round,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[214392   4070]\n",
      " [ 29520   8536]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.98      0.93    218462\n",
      "           1       0.68      0.22      0.34     38056\n",
      "\n",
      "    accuracy                           0.87    256518\n",
      "   macro avg       0.78      0.60      0.63    256518\n",
      "weighted avg       0.85      0.87      0.84    256518\n",
      "\n",
      "Specificity : 0.981369757669526\n",
      "ROC-AUC : 0.6028353938652443\n"
     ]
    }
   ],
   "source": [
    "# evaluate\n",
    "cm = confusion_matrix(y_test, predictions)\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()\n",
    "print(confusion_matrix(y_test, predictions))\n",
    "print(classification_report(y_test, predictions))\n",
    "print(\"Specificity :\", tn/(fp+tn))\n",
    "print(\"ROC-AUC :\", roc_auc_score(y_test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "popular\n",
       "1    215649\n",
       "0    215649\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.value_counts()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyMhz/d4Y547l8Pjx7LlGNZn",
   "collapsed_sections": [
    "ZCahqrluJbR3",
    "DeIjNjA7JhS-",
    "2S2U299QVJdn",
    "5OTHg7coV5pL",
    "R1Bb8VUPJuzX",
    "FxPvvWOGJw1V",
    "uDc3WscPJzfz"
   ],
   "mount_file_id": "13egtzSJN6x69p9vTz4o-zeOvvfGK_hVw",
   "name": "",
   "version": ""
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
