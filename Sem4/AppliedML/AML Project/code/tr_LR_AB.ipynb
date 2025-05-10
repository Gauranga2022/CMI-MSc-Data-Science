{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tmGlRG6lsgTw"
   },
   "source": [
    "##### Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "n_AlSjclkUss"
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "import time\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n",
    "\n",
    "# from gensim.models import Word2Vec\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n",
    "\n",
    "np.set_printoptions(threshold=sys.maxsize)\n",
    "\n",
    "np.random.seed(229)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ingore warnings\n",
    "def warn(*args, **kwargs):\n",
    "    pass\n",
    "import warnings\n",
    "warnings.warn = warn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RNRKcAk2LEXL"
   },
   "source": [
    "### Data Splits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 61847,
     "status": "ok",
     "timestamp": 1745892725361,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "3z1XFX0ckgb5",
    "outputId": "185df43b-1d54-4244-9894-0806555d3526"
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
   "execution_count": 6,
   "metadata": {
    "id": "U-B3l2kAkisk"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1453600, 13), (256518, 13))"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 85% train / 15% test\n",
    "X_train, X_test, y_train, y_test = train_test_split(dat.drop(columns=[\"popular\"]),\n",
    "                                                    dat[\"popular\"],\n",
    "                                                    test_size = 0.15,\n",
    "                                                    random_state = 229)\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "AlMGOkKJkktj"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((430924, 13), (256518, 13))"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# undersample train set\n",
    "majority_size = len(y_train[y_train==0])\n",
    "minority_size = len(y_train[y_train==1])\n",
    "majority_indices = y_train[y_train==0].index\n",
    "rng = np.random.default_rng(seed=229)\n",
    "drop_indices = rng.choice(majority_indices, majority_size-minority_size, replace=False)\n",
    "X_train = X_train.drop(drop_indices)\n",
    "y_train = y_train.drop(drop_indices)\n",
    "\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "popular\n",
       "0    215462\n",
       "1    215462\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "popular\n",
       "0    218275\n",
       "1     38243\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Reviews only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((430924, 1), (256518, 1))"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_review = X_train[[\"tokenized_words\"]]\n",
    "X_test_review = X_test[[\"tokenized_words\"]]\n",
    "\n",
    "X_train_review.shape, X_test_review.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "LOGISTIC REGRESSION BOW\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x11931dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105a2dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106345d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107badd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105e11d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1035c9d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x103ea5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1043b1d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107badd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1046f5d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x103d11d00>\n",
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
      "Training completed in: 1864.50 seconds\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# BAG OF WORDS\n",
    "print(\"\\n\\nLOGISTIC REGRESSION BOW\")\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "# column transorfmer\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('countvectorizer', CountVectorizer(), 'tokenized_words'),  \n",
    "    ], remainder='passthrough'\n",
    ")\n",
    "\n",
    "# full pipeline\n",
    "\n",
    "bow_pipe = make_pipeline(\n",
    "    preprocessor,\n",
    "    LogisticRegression(\n",
    "        penalty='l2',\n",
    "        solver='saga',\n",
    "        max_iter=5000,\n",
    "        random_state=229,\n",
    "        n_jobs=-1\n",
    "    )\n",
    ")\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'logisticregression__C': (10, 1, 0.01, 0.001)\n",
    "}\n",
    "\n",
    "# Set up GridSearchCV\n",
    "gs_bow_pipe = GridSearchCV(\n",
    "    bow_pipe, parameters,\n",
    "    cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=229),\n",
    "    n_jobs=-1)\n",
    "\n",
    "gs_bow_pipe.fit(X_train_review, y_train)\n",
    "\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print(f\"\\nTraining completed in: {total_time:.2f} seconds\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([1737.50078487, 1495.31507421,  169.69934893,  158.93021607]), 'std_fit_time': array([0., 0., 0., 0.]), 'mean_score_time': array([2.2438581 , 3.51356292, 3.67775321, 3.69510198]), 'std_score_time': array([0., 0., 0., 0.]), 'param_logisticregression__C': masked_array(data=[10.0, 1.0, 0.01, 0.001],\n",
      "             mask=[False, False, False, False],\n",
      "       fill_value=1e+20), 'params': [{'logisticregression__C': 10}, {'logisticregression__C': 1}, {'logisticregression__C': 0.01}, {'logisticregression__C': 0.001}], 'split0_test_score': array([0.67074057, 0.67290645, 0.68010025, 0.67278269]), 'mean_test_score': array([0.67074057, 0.67290645, 0.68010025, 0.67278269]), 'std_test_score': array([0., 0., 0., 0.]), 'rank_test_score': array([4, 2, 1, 3], dtype=int32)}\n",
      "{'logisticregression__C': 0.01}\n",
      "\n",
      "Best model saved as 'logistic_bow_model_a.pkl'\n"
     ]
    }
   ],
   "source": [
    "print(gs_bow_pipe.cv_results_)\n",
    "print(gs_bow_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./logistic_bow_model_a.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_bow_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'logistic_bow_model_a.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict\n",
    "predictions = gs_bow_pipe.predict(X_test_review)\n",
    "predictions = list(map(round,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[172536  45739]\n",
      " [ 16723  21520]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.91      0.79      0.85    218275\n",
      "           1       0.32      0.56      0.41     38243\n",
      "\n",
      "    accuracy                           0.76    256518\n",
      "   macro avg       0.62      0.68      0.63    256518\n",
      "weighted avg       0.82      0.76      0.78    256518\n",
      "\n",
      "Specificity : 0.7904524109494904\n",
      "ROC-AUC : 0.676584885494618\n"
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
    "id": "raupEfc6ruUV"
   },
   "source": [
    "### Review Meta Data Only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((430924, 12), (256518, 12))"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_no_tokens = X_train.drop(columns=[\"tokenized_words\"])\n",
    "X_test_no_tokens = X_test.drop(columns=[\"tokenized_words\"])\n",
    "\n",
    "X_train_no_tokens.shape, X_test_no_tokens.shape"
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
     "elapsed": 10863793,
     "status": "ok",
     "timestamp": 1745878626981,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "FUdfW-T-koqQ",
    "outputId": "b3dc918c-c8d1-411e-ff68-cfd96a0d64b1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      "LOGISTIC REGRESSION BOW\n",
      "\n",
      "Training completed in: 36.84 seconds\n",
      "\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106f1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106c1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106a49d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10631dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10d01dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104455d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x106c1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10551dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10a11dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1032d1d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x10d01dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x105f55d00>\n",
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
    "print(\"\\n\\nLOGISTIC REGRESSION BOW\")\n",
    "\n",
    "numerical_cols = [col for col in X_train.columns if col != 'tokenized_words']\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "# column transorfmer\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('standardscaler', StandardScaler(), numerical_cols)       # Scale numerical columns\n",
    "    ], remainder='passthrough'\n",
    ")\n",
    "\n",
    "# full pipeline\n",
    "\n",
    "bow_pipe = make_pipeline(\n",
    "    preprocessor,\n",
    "    LogisticRegression(\n",
    "        penalty='l2',\n",
    "        solver='saga',\n",
    "        max_iter=5000,\n",
    "        random_state=229,\n",
    "        n_jobs=-1\n",
    "    )\n",
    ")\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'logisticregression__C': (10, 1, 0.01, 0.001)\n",
    "}\n",
    "\n",
    "\n",
    "# Set up GridSearchCV\n",
    "gs_bow_pipe = GridSearchCV(\n",
    "    bow_pipe, parameters,\n",
    "    cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=229),\n",
    "    n_jobs=-1)\n",
    "\n",
    "gs_bow_pipe.fit(X_train_no_tokens, y_train)\n",
    "\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print(f\"\\nTraining completed in: {total_time:.2f} seconds\\n\\n\")"
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
      "{'mean_fit_time': array([19.61022806, 19.6353538 , 18.77423787, 13.76602197]), 'std_fit_time': array([0., 0., 0., 0.]), 'mean_score_time': array([0.00911593, 0.00812221, 0.00802708, 0.00793099]), 'std_score_time': array([0., 0., 0., 0.]), 'param_logisticregression__C': masked_array(data=[10.0, 1.0, 0.01, 0.001],\n",
      "             mask=[False, False, False, False],\n",
      "       fill_value=1e+20), 'params': [{'logisticregression__C': 10}, {'logisticregression__C': 1}, {'logisticregression__C': 0.01}, {'logisticregression__C': 0.001}], 'split0_test_score': array([0.67917202, 0.67917202, 0.67886261, 0.67866149]), 'mean_test_score': array([0.67917202, 0.67917202, 0.67886261, 0.67866149]), 'std_test_score': array([0., 0., 0., 0.]), 'rank_test_score': array([1, 1, 3, 4], dtype=int32)}\n",
      "{'logisticregression__C': 10}\n",
      "\n",
      "Best model saved as 'logistic_bow_model_b.pkl'\n"
     ]
    }
   ],
   "source": [
    "print(gs_bow_pipe.cv_results_)\n",
    "print(gs_bow_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./logistic_bow_model_b.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_bow_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'logistic_bow_model_b.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# predict\n",
    "predictions = gs_bow_pipe.predict(X_test_no_tokens)\n",
    "predictions = list(map(round,predictions))"
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
     "elapsed": 29284,
     "status": "ok",
     "timestamp": 1745878656082,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "967VNz3bs9Xp",
    "outputId": "dc7093f1-0500-4907-f37b-89247eb639cb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[167068  51207]\n",
      " [ 15515  22728]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.92      0.77      0.83    218275\n",
      "           1       0.31      0.59      0.41     38243\n",
      "\n",
      "    accuracy                           0.74    256518\n",
      "   macro avg       0.61      0.68      0.62    256518\n",
      "weighted avg       0.82      0.74      0.77    256518\n",
      "\n",
      "Specificity : 0.7654014431336617\n",
      "ROC-AUC : 0.6798531416175592\n"
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
   "metadata": {},
   "source": [
    "### Diagnostics "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     actual  predicted\n",
      "user_reviews       0.266226   0.553409\n",
      "days_since_review -0.022886  -0.042072\n",
      "user_rating       -0.019622  -0.064470\n",
      "rating_diff       -0.007004  -0.026239\n",
      "num_words          0.203718   0.573106\n",
      "avg_word_len       0.034796   0.089160\n",
      "avg_sent_len       0.057920   0.170383\n",
      "pct_verbs         -0.030523  -0.066526\n",
      "pct_nouns          0.018215   0.035825\n",
      "pct_adj           -0.046640  -0.127933\n",
      "quote              0.119136   0.352901\n",
      "sentiment         -0.049510  -0.142077\n",
      "actual             1.000000   0.282860\n",
      "predicted          0.282860   1.000000\n"
     ]
    }
   ],
   "source": [
    "corr_df = X_test_no_tokens.copy()\n",
    "corr_df[\"actual\"] = y_test.values\n",
    "corr_df[\"predicted\"] = predictions\n",
    "\n",
    "correlations = corr_df.corr(numeric_only=True)  # Only numerical columns\n",
    "\n",
    "print(correlations[[\"actual\", \"predicted\"]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Review Meta Data + Full sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1453600, 13), (256518, 13))"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# read/prep data\n",
    "dat = pd.read_csv(\"../data/tokenized_reviews.csv\")\n",
    "dat = dat.dropna()\n",
    "dat[\"quote\"] = dat[\"quote\"].astype(int)\n",
    "dat[\"tokenized_words\"] = dat[\"tokenized_words\"].apply(lambda x: x.strip(\"[']\").replace(\"', '\",\" \"))\n",
    "\n",
    "# 85% train / 15% test\n",
    "X_train, X_test, y_train, y_test = train_test_split(dat.drop(columns=[\"popular\"]),\n",
    "                                                    dat[\"popular\"],\n",
    "                                                    test_size = 0.15,\n",
    "                                                    random_state = 229)\n",
    "X_train.shape, X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((1453600, 12), (256518, 12))"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_no_tokens = X_train.drop(columns=[\"tokenized_words\"])\n",
    "X_test_no_tokens = X_test.drop(columns=[\"tokenized_words\"])\n",
    "\n",
    "X_train_no_tokens.shape, X_test_no_tokens.shape"
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
      "\n",
      "\n",
      "LOGISTIC REGRESSION BOW\n",
      "\n",
      "Training completed in: 121.47 seconds\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# BAG OF WORDS\n",
    "print(\"\\n\\nLOGISTIC REGRESSION BOW\")\n",
    "\n",
    "numerical_cols = [col for col in X_train.columns if col != 'tokenized_words']\n",
    "\n",
    "start_time = time.time()\n",
    "\n",
    "# column transorfmer\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('standardscaler', StandardScaler(), numerical_cols)       # Scale numerical columns\n",
    "    ], remainder='passthrough'\n",
    ")\n",
    "\n",
    "# full pipeline\n",
    "\n",
    "bow_pipe = make_pipeline(\n",
    "    preprocessor,\n",
    "    LogisticRegression(\n",
    "        penalty='l2',\n",
    "        solver='saga',\n",
    "        max_iter=5000,\n",
    "        random_state=229,\n",
    "        n_jobs=-1\n",
    "    )\n",
    ")\n",
    "\n",
    "# parameters to try\n",
    "parameters = {\n",
    "    'logisticregression__C': (10, 1, 0.01, 0.001)\n",
    "}\n",
    "\n",
    "\n",
    "# Set up GridSearchCV\n",
    "gs_bow_f_pipe = GridSearchCV(\n",
    "    bow_pipe, parameters,\n",
    "    cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=229),\n",
    "    n_jobs=-1)\n",
    "\n",
    "gs_bow_f_pipe.fit(X_train_no_tokens, y_train)\n",
    "\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print(f\"\\nTraining completed in: {total_time:.2f} seconds\\n\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'mean_fit_time': array([60.96135998, 62.30540109, 60.23921776, 50.71277094]), 'std_fit_time': array([0., 0., 0., 0.]), 'mean_score_time': array([0.02221513, 0.01952696, 0.01981711, 0.02140498]), 'std_score_time': array([0., 0., 0., 0.]), 'param_logisticregression__C': masked_array(data=[10.0, 1.0, 0.01, 0.001],\n",
      "             mask=[False, False, False, False],\n",
      "       fill_value=1e+20), 'params': [{'logisticregression__C': 10}, {'logisticregression__C': 1}, {'logisticregression__C': 0.01}, {'logisticregression__C': 0.001}], 'split0_test_score': array([0.85258668, 0.85258668, 0.85260503, 0.85256375]), 'mean_test_score': array([0.85258668, 0.85258668, 0.85260503, 0.85256375]), 'std_test_score': array([0., 0., 0., 0.]), 'rank_test_score': array([2, 2, 1, 4], dtype=int32)}\n",
      "{'logisticregression__C': 0.01}\n",
      "\n",
      "Best model saved as 'logistic_bow_model_fb.pkl'\n"
     ]
    }
   ],
   "source": [
    "print(gs_bow_f_pipe.cv_results_)\n",
    "print(gs_bow_f_pipe.best_params_)\n",
    "\n",
    "# save the best model with pickle\n",
    "with open(\"./logistic_bow_model_fb.pkl\", \"wb\") as f:\n",
    "    pickle.dump(gs_bow_f_pipe.best_estimator_, f)\n",
    "\n",
    "print(\"\\nBest model saved as 'logistic_bow_model_fb.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[215286   2989]\n",
      " [ 35149   3094]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.99      0.92    218275\n",
      "           1       0.51      0.08      0.14     38243\n",
      "\n",
      "    accuracy                           0.85    256518\n",
      "   macro avg       0.68      0.53      0.53    256518\n",
      "weighted avg       0.81      0.85      0.80    256518\n",
      "\n",
      "Specificity : 0.9863062650326423\n",
      "ROC-AUC : 0.5336049799132304\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107b1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107e1dd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1073fdd00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x107585d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x104c49d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x103209d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x102d61d00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n",
      "Exception ignored in: <function ResourceTracker.__del__ at 0x1043add00>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 77, in __del__\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 86, in _stop\n",
      "  File \"/opt/homebrew/Cellar/python@3.12/3.12.10/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py\", line 111, in _stop_locked\n",
      "ChildProcessError: [Errno 10] No child processes\n"
     ]
    }
   ],
   "source": [
    "# predict\n",
    "predictions = gs_bow_f_pipe.predict(X_test_no_tokens)\n",
    "predictions = list(map(round,predictions))\n",
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "RNRKcAk2LEXL"
   ],
   "provenance": []
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
