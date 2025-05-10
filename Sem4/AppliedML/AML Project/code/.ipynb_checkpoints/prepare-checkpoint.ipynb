{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BVeN1H25q2Tf"
   },
   "source": [
    "---\n",
    "##### What does this notebook prepare.py` do? – cleaning and labelling\n",
    "\n",
    "**Big idea**   \n",
    "Start with two enormous JSON files (15.7 M reviews + 2.3 M books) and boil them down to a tidy CSV of *English* reviews that have enough reader activity to measure “popularity”.\n",
    "\n",
    "**Step-by-step**\n",
    "\n",
    "1. **Keep only engaged reviews** – Skip any review whose `n_votes + n_comments` is 0.   \n",
    "   *Example* : A review with 3 likes and 1 comment stays; a review with 0 likes, 0 comments is discarded.\n",
    "2. **Remove quiet books** – If a book has fewer than **10 total reviews** **or** fewer than **60 combined likes+comments** across all its reviews, throw all its reviews away.  This ensures we can judge “popular within its crowd”.\n",
    "3. **English-only filter** – Run Facebook’s fastText language-ID model.  Keep only rows predicted `__label__en` with probability ≥ 0.9.  Roughly 84 % survive.\n",
    "4. **Create helper columns**\n",
    "   * `review_likes` = likes + comments  \n",
    "   * `book_review_likes` = total likes/comments for that book  \n",
    "   * `like_share` = `review_likes` ÷ `book_review_likes`\n",
    "5. **Binary target** – `popular = 1` if `like_share > 0.02`, else 0.  (≈ 21 % positives).\n",
    "6. **Days since review** – Convert Goodreads timestamp to datetime and compute how many days have passed.  \n",
    "7. **Count reviews per user** – `user_reviews` column (experience proxy).  \n",
    "8. **Save** – dump everything to `filtered_reviews.csv`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 45928,
     "status": "ok",
     "timestamp": 1745697642336,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "L0b3OBvKqNBE",
    "outputId": "6ddc7c6d-3d52-42d2-adcf-e346c71e9a08"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fasttext\n",
      "  Downloading fasttext-0.9.3.tar.gz (73 kB)\n",
      "  Installing build dependencies ... \u001b[done\n",
      "\u001b[?25h  Getting requirements to build wheel ... \u001b[?25ldone\n",
      "\u001b[?25h  Preparing metadata (pyproject.toml) ... \u001b[?25ldone\n",
      "\u001b[?25hCollecting pybind11>=2.2 (from fasttext)\n",
      "  Using cached pybind11-2.13.6-py3-none-any.whl.metadata (9.5 kB)\n",
      "Requirement already satisfied: setuptools>=0.7.0 in /Users/vasu/AjrVasu/Coding/iit/degree/AppliedMachineLearning/envaml/lib/python3.12/site-packages (from fasttext) (78.1.0)\n",
      "Requirement already satisfied: numpy in /Users/vasu/AjrVasu/Coding/iit/degree/AppliedMachineLearning/envaml/lib/python3.12/site-packages (from fasttext) (2.2.2)\n",
      "Using cached pybind11-2.13.6-py3-none-any.whl (243 kB)\n",
      "Building wheels for collected packages: fasttext\n",
      "  Building wheel for fasttexdone\n",
      "\u001b[?25h  Created wheel for fasttext: filename=fasttext-0.9.3-cp312-cp312-macosx_15_0_arm64.whl size=307504 sha256=41d53fc323a7e884371a68b2b45ea403fbeb3184f60c0232f016da8c00a7d8a6\n",
      "  Stored in directory: /Users/vasu/Library/Caches/pip/wheels/20/27/95/a7baf1b435f1cbde017cabdf1e9688526d2b0e929255a359c6\n",
      "Successfully built fasttext\n",
      "Installing collected packages: pybind11, fasttext\n",
      "Successfully installed fasttext-0.9.3 pybind11-2.13.6\n",
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m25.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m25.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "!pip install fasttext"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "VbSLSmA2skx2"
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import random\n",
    "import urllib.request\n",
    "import gzip\n",
    "import matplotlib.pyplot as plt \n",
    "import json\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import fasttext"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FwA0iZYou1J3"
   },
   "source": [
    "##### Parse the files and read the number of books/reviews"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 369502,
     "status": "ok",
     "timestamp": 1745698073424,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "PCzrVDDxNyw5",
    "outputId": "3fde728e-7a39-4ad5-a34f-5bc99489e996"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total_reviews:  15739967\n"
     ]
    }
   ],
   "source": [
    "# updated paths to match your Drive\n",
    "review_path = \"../data/goodreads_reviews_dedup.json.gz\"\n",
    "book_path = \"../data/goodreads_books.json.gz\"\n",
    "\n",
    "# read review data, removing entries with 0 likes\n",
    "reviews = []\n",
    "total_reviews = 0\n",
    "with gzip.open(review_path, 'rt', encoding='utf-8') as f:\n",
    "    for line in f:\n",
    "        total_reviews += 1\n",
    "        entry = json.loads(line)\n",
    "        if entry['n_votes'] + entry['n_comments'] > 0:\n",
    "            reviews.append(entry)\n",
    "\n",
    "print(\"total_reviews: \", total_reviews)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'user_id': '8842281e1d1347389f2ab93d60773d4d',\n",
       " 'book_id': '24375664',\n",
       " 'review_id': '5cd416f3efc3f944fce4ce2db2290d5e',\n",
       " 'rating': 5,\n",
       " 'review_text': \"Mind blowingly cool. Best science fiction I've read in some time. I just loved all the descriptions of the society of the future - how they lived in trees, the notion of owning property or even getting married was gone. How every surface was a screen. \\n The undulations of how society responds to the Trisolaran threat seem surprising to me. Maybe its more the Chinese perspective, but I wouldn't have thought the ETO would exist in book 1, and I wouldn't have thought people would get so over-confident in our primitive fleet's chances given you have to think that with superior science they would have weapons - and defenses - that would just be as rifles to arrows once were. \\n But the moment when Luo Ji won as a wallfacer was just too cool. I may have actually done a fist pump. Though by the way, if the Dark Forest theory is right - and I see no reason why it wouldn't be - we as a society should probably stop broadcasting so much signal out into the universe.\",\n",
       " 'date_added': 'Fri Aug 25 13:55:02 -0700 2017',\n",
       " 'date_updated': 'Mon Oct 09 08:55:59 -0700 2017',\n",
       " 'read_at': 'Sat Oct 07 00:00:00 -0700 2017',\n",
       " 'started_at': 'Sat Aug 26 00:00:00 -0700 2017',\n",
       " 'n_votes': 16,\n",
       " 'n_comments': 0}"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reviews[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "reviews with 1+ like/comment:  4706961\n"
     ]
    }
   ],
   "source": [
    "# convert to DF, retaining only relevant columns\n",
    "reviews = pd.DataFrame(reviews, columns=['user_id','book_id','rating','review_text','date_added','n_votes','n_comments'])\n",
    "print(\"reviews with 1+ like/comment: \", len(reviews))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>review_text</th>\n",
       "      <th>date_added</th>\n",
       "      <th>n_votes</th>\n",
       "      <th>n_comments</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>24375664</td>\n",
       "      <td>5</td>\n",
       "      <td>Mind blowingly cool. Best science fiction I've...</td>\n",
       "      <td>Fri Aug 25 13:55:02 -0700 2017</td>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>5</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>Sun Jul 30 07:44:10 -0700 2017</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6392944</td>\n",
       "      <td>3</td>\n",
       "      <td>I haven't read a fun mystery book in a while a...</td>\n",
       "      <td>Mon Jul 24 02:48:17 -0700 2017</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>22078596</td>\n",
       "      <td>4</td>\n",
       "      <td>Fun, fast paced, and disturbing tale of murder...</td>\n",
       "      <td>Mon Jul 24 02:33:09 -0700 2017</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6644782</td>\n",
       "      <td>4</td>\n",
       "      <td>A fun book that gives you a sense of living in...</td>\n",
       "      <td>Mon Jul 24 02:28:14 -0700 2017</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  24375664       5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d  18245960       5   \n",
       "2  8842281e1d1347389f2ab93d60773d4d   6392944       3   \n",
       "3  8842281e1d1347389f2ab93d60773d4d  22078596       4   \n",
       "4  8842281e1d1347389f2ab93d60773d4d   6644782       4   \n",
       "\n",
       "                                         review_text  \\\n",
       "0  Mind blowingly cool. Best science fiction I've...   \n",
       "1  This is a special book. It started slow for ab...   \n",
       "2  I haven't read a fun mystery book in a while a...   \n",
       "3  Fun, fast paced, and disturbing tale of murder...   \n",
       "4  A fun book that gives you a sense of living in...   \n",
       "\n",
       "                       date_added  n_votes  n_comments  \n",
       "0  Fri Aug 25 13:55:02 -0700 2017       16           0  \n",
       "1  Sun Jul 30 07:44:10 -0700 2017       28           1  \n",
       "2  Mon Jul 24 02:48:17 -0700 2017        6           0  \n",
       "3  Mon Jul 24 02:33:09 -0700 2017       22           4  \n",
       "4  Mon Jul 24 02:28:14 -0700 2017        8           0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reviews.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "user_id\n",
       "00000377eea48021d3002730d56aca9a     1\n",
       "00009ab2ed8cbfceda5a59da40966321     1\n",
       "00009e46d18f223a82b22da38586b605    11\n",
       "0000c3d51aa099745e93a4e99c4856c8    68\n",
       "0001085188e302fc6b2568de45a5f56b     2\n",
       "Name: user_reviews, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# calculate number of reviews per user\n",
    "user_reviews = reviews.groupby(\"user_id\")[\"book_id\"].count().rename(\"user_reviews\")\n",
    "user_reviews.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>review_text</th>\n",
       "      <th>date_added</th>\n",
       "      <th>n_votes</th>\n",
       "      <th>n_comments</th>\n",
       "      <th>review_likes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>24375664</td>\n",
       "      <td>5</td>\n",
       "      <td>Mind blowingly cool. Best science fiction I've...</td>\n",
       "      <td>Fri Aug 25 13:55:02 -0700 2017</td>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>5</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>Sun Jul 30 07:44:10 -0700 2017</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6392944</td>\n",
       "      <td>3</td>\n",
       "      <td>I haven't read a fun mystery book in a while a...</td>\n",
       "      <td>Mon Jul 24 02:48:17 -0700 2017</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>22078596</td>\n",
       "      <td>4</td>\n",
       "      <td>Fun, fast paced, and disturbing tale of murder...</td>\n",
       "      <td>Mon Jul 24 02:33:09 -0700 2017</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6644782</td>\n",
       "      <td>4</td>\n",
       "      <td>A fun book that gives you a sense of living in...</td>\n",
       "      <td>Mon Jul 24 02:28:14 -0700 2017</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  24375664       5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d  18245960       5   \n",
       "2  8842281e1d1347389f2ab93d60773d4d   6392944       3   \n",
       "3  8842281e1d1347389f2ab93d60773d4d  22078596       4   \n",
       "4  8842281e1d1347389f2ab93d60773d4d   6644782       4   \n",
       "\n",
       "                                         review_text  \\\n",
       "0  Mind blowingly cool. Best science fiction I've...   \n",
       "1  This is a special book. It started slow for ab...   \n",
       "2  I haven't read a fun mystery book in a while a...   \n",
       "3  Fun, fast paced, and disturbing tale of murder...   \n",
       "4  A fun book that gives you a sense of living in...   \n",
       "\n",
       "                       date_added  n_votes  n_comments  review_likes  \n",
       "0  Fri Aug 25 13:55:02 -0700 2017       16           0            16  \n",
       "1  Sun Jul 30 07:44:10 -0700 2017       28           1            29  \n",
       "2  Mon Jul 24 02:48:17 -0700 2017        6           0             6  \n",
       "3  Mon Jul 24 02:33:09 -0700 2017       22           4            26  \n",
       "4  Mon Jul 24 02:28:14 -0700 2017        8           0             8  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# combine votes and comments\n",
    "reviews[\"review_likes\"] = reviews[\"n_votes\"] + reviews[\"n_comments\"]\n",
    "# reviews = reviews.drop([\"n_votes\",\"n_comments\"],axis=1)\n",
    "reviews.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# read book data, removing books with fewer than 10 reviews\n",
    "books = []\n",
    "total_books = 0\n",
    "with gzip.open(book_path, 'rt', encoding='utf-8') as f:  # <-- gzip.open + 'rt' mode here too\n",
    "    for line in f:\n",
    "        total_books += 1\n",
    "        entry = json.loads(line)\n",
    "        try:\n",
    "            if int(entry['text_reviews_count']) > 0:\n",
    "                books.append(entry)\n",
    "        except:\n",
    "            pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total_books:  2360655\n",
      "shortlisted books:  2359404\n"
     ]
    }
   ],
   "source": [
    "print(\"total_books: \", total_books)\n",
    "print(\"shortlisted books: \", len(books))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'isbn': '1499346603', 'text_reviews_count': '1', 'series': [], 'country_code': 'US', 'language_code': 'eng', 'popular_shelves': [], 'asin': '', 'is_ebook': 'false', 'average_rating': '5.00', 'kindle_asin': '', 'similar_books': [], 'description': 'R. Jack Winter\\'s debut novel, GANI & SEAN follows two assassins whose paths cross when one fails to fulfill her contract to kill the other. Sean LePen misses her first shot when she realizes her target, Gani Jaksic is a man she admires for years, since her days in weapons training under his tutelage. When her boss Kristoff Koczella - the brutal and bizarre Chicago kingpin - discovers his assassin\\'s mistake, he hires a highly skilled hit man known as \"the voice\" to kill her and her target, Gani.\\nSean flees across country with both Gani and \"the voice\" on her trail. Will she escape? Will Gani?', 'format': 'Paperback', 'link': 'https://www.goodreads.com/book/show/22468647-gani-sean', 'authors': [{'author_id': '8305679', 'role': ''}], 'publisher': 'R. Jack Winter', 'num_pages': '284', 'publication_day': '', 'isbn13': '9781499346602', 'publication_month': '', 'edition_information': '', 'publication_year': '2014', 'url': 'https://www.goodreads.com/book/show/22468647-gani-sean', 'image_url': 'https://images.gr-assets.com/books/1403203069m/22468647.jpg', 'book_id': '22468647', 'ratings_count': '1', 'work_id': '41908423', 'title': 'Gani & Sean', 'title_without_series': 'Gani & Sean'}\n"
     ]
    }
   ],
   "source": [
    "random_books = random.sample(books, 1)\n",
    "for book in random_books:\n",
    "    print(book)"
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
     "elapsed": 62,
     "status": "ok",
     "timestamp": 1745698214916,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "Mi2AgVO6uFEw",
    "outputId": "9233df07-c19b-4e56-be2c-a88468f98c62"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total_reviews:  15739967\n",
      "reviews with 1+ like/comment:  4706961\n",
      "total_books:  2360655\n"
     ]
    }
   ],
   "source": [
    "print(\"total_reviews: \", total_reviews)\n",
    "print(\"reviews with 1+ like/comment: \", len(reviews))\n",
    "print(\"total_books: \", total_books)"
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
      "books with 10+ reviews:  2359404\n"
     ]
    }
   ],
   "source": [
    "# convert to DF, retaining only relevant columns\n",
    "books = pd.DataFrame(books, columns=[\"book_id\",\"text_reviews_count\",\"ratings_count\",\"average_rating\"])\n",
    "print(\"books with 10+ reviews: \", len(books))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>book_id</th>\n",
       "      <th>text_reviews_count</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>average_rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5333265</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1333909</td>\n",
       "      <td>6</td>\n",
       "      <td>10</td>\n",
       "      <td>3.23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7327624</td>\n",
       "      <td>7</td>\n",
       "      <td>140</td>\n",
       "      <td>4.03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6066819</td>\n",
       "      <td>3282</td>\n",
       "      <td>51184</td>\n",
       "      <td>3.49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>287140</td>\n",
       "      <td>5</td>\n",
       "      <td>15</td>\n",
       "      <td>3.40</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   book_id text_reviews_count ratings_count average_rating\n",
       "0  5333265                  1             3           4.00\n",
       "1  1333909                  6            10           3.23\n",
       "2  7327624                  7           140           4.03\n",
       "3  6066819               3282         51184           3.49\n",
       "4   287140                  5            15           3.40"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "books.head()"
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
      "join reviews/books:  4706729\n"
     ]
    }
   ],
   "source": [
    "# join reviews and books\n",
    "dat = pd.merge(reviews,books,on=\"book_id\")\n",
    "print(\"join reviews/books: \", len(dat))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>review_text</th>\n",
       "      <th>date_added</th>\n",
       "      <th>n_votes</th>\n",
       "      <th>n_comments</th>\n",
       "      <th>review_likes</th>\n",
       "      <th>text_reviews_count</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>average_rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>24375664</td>\n",
       "      <td>5</td>\n",
       "      <td>Mind blowingly cool. Best science fiction I've...</td>\n",
       "      <td>Fri Aug 25 13:55:02 -0700 2017</td>\n",
       "      <td>16</td>\n",
       "      <td>0</td>\n",
       "      <td>16</td>\n",
       "      <td>154</td>\n",
       "      <td>2925</td>\n",
       "      <td>4.38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>5</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>Sun Jul 30 07:44:10 -0700 2017</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>374</td>\n",
       "      <td>6336</td>\n",
       "      <td>4.01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6392944</td>\n",
       "      <td>3</td>\n",
       "      <td>I haven't read a fun mystery book in a while a...</td>\n",
       "      <td>Mon Jul 24 02:48:17 -0700 2017</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "      <td>42</td>\n",
       "      <td>675</td>\n",
       "      <td>3.80</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>22078596</td>\n",
       "      <td>4</td>\n",
       "      <td>Fun, fast paced, and disturbing tale of murder...</td>\n",
       "      <td>Mon Jul 24 02:33:09 -0700 2017</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>26</td>\n",
       "      <td>28</td>\n",
       "      <td>429</td>\n",
       "      <td>4.00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>6644782</td>\n",
       "      <td>4</td>\n",
       "      <td>A fun book that gives you a sense of living in...</td>\n",
       "      <td>Mon Jul 24 02:28:14 -0700 2017</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>8</td>\n",
       "      <td>98</td>\n",
       "      <td>3.76</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  24375664       5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d  18245960       5   \n",
       "2  8842281e1d1347389f2ab93d60773d4d   6392944       3   \n",
       "3  8842281e1d1347389f2ab93d60773d4d  22078596       4   \n",
       "4  8842281e1d1347389f2ab93d60773d4d   6644782       4   \n",
       "\n",
       "                                         review_text  \\\n",
       "0  Mind blowingly cool. Best science fiction I've...   \n",
       "1  This is a special book. It started slow for ab...   \n",
       "2  I haven't read a fun mystery book in a while a...   \n",
       "3  Fun, fast paced, and disturbing tale of murder...   \n",
       "4  A fun book that gives you a sense of living in...   \n",
       "\n",
       "                       date_added  n_votes  n_comments  review_likes  \\\n",
       "0  Fri Aug 25 13:55:02 -0700 2017       16           0            16   \n",
       "1  Sun Jul 30 07:44:10 -0700 2017       28           1            29   \n",
       "2  Mon Jul 24 02:48:17 -0700 2017        6           0             6   \n",
       "3  Mon Jul 24 02:33:09 -0700 2017       22           4            26   \n",
       "4  Mon Jul 24 02:28:14 -0700 2017        8           0             8   \n",
       "\n",
       "  text_reviews_count ratings_count average_rating  \n",
       "0                154          2925           4.38  \n",
       "1                374          6336           4.01  \n",
       "2                 42           675           3.80  \n",
       "3                 28           429           4.00  \n",
       "4                  8            98           3.76  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HiPCdHo-uvEU"
   },
   "source": [
    "##### Download the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 1696,
     "status": "ok",
     "timestamp": 1745698223487,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "EKJpWGmwuBdr",
    "outputId": "5d65632d-a9df-4352-8843-c34ab6cc2551"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model not found, downloading...\n",
      "Download complete.\n"
     ]
    }
   ],
   "source": [
    "# Model filename\n",
    "model_path = 'lid.176.bin'\n",
    "\n",
    "# Check if the model file exists\n",
    "if not os.path.exists(model_path):\n",
    "    print(\"Model not found, downloading...\")\n",
    "    url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'\n",
    "    urllib.request.urlretrieve(url, model_path)\n",
    "    print(\"Download complete.\")\n",
    "\n",
    "# Load the model\n",
    "language_model = fasttext.load_model(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "review_likes\n",
       "1       2176483\n",
       "2        852482\n",
       "3        444386\n",
       "4        269989\n",
       "5        177754\n",
       "         ...   \n",
       "580           1\n",
       "1198          1\n",
       "527           1\n",
       "1124          1\n",
       "862           1\n",
       "Name: count, Length: 882, dtype: int64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat['review_likes'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "reviews on books with 60+ review likes:  2027220\n"
     ]
    }
   ],
   "source": [
    "# calculate total review likes per book\n",
    "book_review_likes = dat.groupby(\"book_id\")[\"review_likes\"].sum().rename(\"book_review_likes\")\n",
    "\n",
    "# remove books with fewer than 100 total likes on reviews\n",
    "book_review_likes = book_review_likes[book_review_likes>=100]\n",
    "dat = dat.merge(book_review_likes,on='book_id')\n",
    "print(\"reviews on books with 100+ review likes: \", len(dat))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>review_text</th>\n",
       "      <th>date_added</th>\n",
       "      <th>n_votes</th>\n",
       "      <th>n_comments</th>\n",
       "      <th>review_likes</th>\n",
       "      <th>text_reviews_count</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>average_rating</th>\n",
       "      <th>book_review_likes</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>5</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>Sun Jul 30 07:44:10 -0700 2017</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>374</td>\n",
       "      <td>6336</td>\n",
       "      <td>4.01</td>\n",
       "      <td>199</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>16981</td>\n",
       "      <td>3</td>\n",
       "      <td>Recommended by Don Katz. Avail for free in Dec...</td>\n",
       "      <td>Mon Dec 05 10:46:44 -0800 2016</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3741</td>\n",
       "      <td>125232</td>\n",
       "      <td>3.84</td>\n",
       "      <td>631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>28684704</td>\n",
       "      <td>3</td>\n",
       "      <td>A fun, fast paced science fiction thriller. I ...</td>\n",
       "      <td>Tue Nov 15 11:29:22 -0800 2016</td>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>22</td>\n",
       "      <td>1026</td>\n",
       "      <td>13990</td>\n",
       "      <td>4.10</td>\n",
       "      <td>1010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>27161156</td>\n",
       "      <td>0</td>\n",
       "      <td>Recommended reading to understand what is goin...</td>\n",
       "      <td>Wed Nov 09 17:37:04 -0800 2016</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>13663</td>\n",
       "      <td>99022</td>\n",
       "      <td>3.96</td>\n",
       "      <td>4340</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>32283133</td>\n",
       "      <td>0</td>\n",
       "      <td>http://www.telegraph.co.uk/culture/10...</td>\n",
       "      <td>Tue Nov 01 11:09:18 -0700 2016</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>3061</td>\n",
       "      <td>17494</td>\n",
       "      <td>3.86</td>\n",
       "      <td>1436</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  18245960       5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d     16981       3   \n",
       "2  8842281e1d1347389f2ab93d60773d4d  28684704       3   \n",
       "3  8842281e1d1347389f2ab93d60773d4d  27161156       0   \n",
       "4  8842281e1d1347389f2ab93d60773d4d  32283133       0   \n",
       "\n",
       "                                         review_text  \\\n",
       "0  This is a special book. It started slow for ab...   \n",
       "1  Recommended by Don Katz. Avail for free in Dec...   \n",
       "2  A fun, fast paced science fiction thriller. I ...   \n",
       "3  Recommended reading to understand what is goin...   \n",
       "4           http://www.telegraph.co.uk/culture/10...   \n",
       "\n",
       "                       date_added  n_votes  n_comments  review_likes  \\\n",
       "0  Sun Jul 30 07:44:10 -0700 2017       28           1            29   \n",
       "1  Mon Dec 05 10:46:44 -0800 2016        1           0             1   \n",
       "2  Tue Nov 15 11:29:22 -0800 2016       22           0            22   \n",
       "3  Wed Nov 09 17:37:04 -0800 2016        5           1             6   \n",
       "4  Tue Nov 01 11:09:18 -0700 2016        9           0             9   \n",
       "\n",
       "  text_reviews_count ratings_count average_rating  book_review_likes  \n",
       "0                374          6336           4.01                199  \n",
       "1               3741        125232           3.84                631  \n",
       "2               1026         13990           4.10               1010  \n",
       "3              13663         99022           3.96               4340  \n",
       "4               3061         17494           3.86               1436  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABIsAAALLCAYAAAB5BwI4AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA6+lJREFUeJzs3Qd0HcX5sPGxsQGDAdM7mA6mF0PovXcCoYbeEvFRTDMltNB7ABEgtNBCDRAgtAABQhW9iF5C770X3+8882c2q/WVLNuytHv9/M4R2FfX0s6d3dnZd96Z6VWr1WpBkiRJkiRJCiH07ukDkCRJkiRJUnkYLJIkSZIkSVLGYJEkSZIkSZIyBoskSZIkSZKUMVgkSZIkSZKkjMEiSZIkSZIkZQwWSZIkSZIkKWOwSJIkSZIkSRmDRZIkSZIkScoYLJKkkjjssMNCr169uuV3rbDCCvEr+fe//x1/99VXX90tv3/bbbcNAwcODGX21VdfhR133DFMM8008bPZc889QxnxOfJ5VkHxWC+88ML42T7yyCOhDPU91VRThUsvvXSU/v3rr78ey0KZyix95hxvWdojhR5rizsydOjQsMQSS3T6/Rx3+jrxxBPH6LFVEZ8L19XIeuKJJ9p8tmU4NySNHQwWSdIYfCBLX+OPP36Ybrrpwuqrrx5OO+208OWXX3bJ73nnnXdi55POZNmU+dg64+ijj471+Lvf/S5cfPHF4be//W2HQZB8fU844YRh8cUXDxdddFEYG6RASZUfEP/0pz+FiSaaKGy22WbDBUzSV+/evcO0004b1llnnfDggw/26PGW2f333x8/u88++6xLf+4dd9wRtt9++zDnnHOGCSaYIMw666wxoPvuu++2exzLLLNMfC9B39133z0GBTt7Pue/Jp544rDQQguFM844I/z888+hKs4888xRDmASIH/yySfDP/7xj07/mw033DC2l2uvvXYpA2AYNmxYbJsJhE022WTxuuec2nrrrUt5Xc8888zxMz3wwAN7+lAkjWX69PQBSFIjO+KII8Iss8wSfvzxx/Dee+/FTjMd8JNPPjl2wBdYYIHsvQcffHAcyR3ZgMzhhx8egxU8yHTWbbfdFsa0jo7tL3/5S+ywl9mdd94ZfvWrX4VDDz20U++njHvvvXf8Mw+v5557bthmm23C999/H3baaacxdpwvvPBCDGJUQVmPleuTYNFee+0VxhlnnOG+/+c//zn0798/nrNvvvlmPH+XW2658PDDD2fnNg903377bejbt29oBKPSHuWDNFz7ZJENGDCgy45p//33D5988knYZJNNwhxzzBFeffXVGLy58cYbY1CagFDC31deeeUwzzzzxPb2rbfeisHMl156Kdx8882d+n2bb755WGutteKfP//88/DPf/4z/L//9//Cf//733DCCSeEqgSLpphiilHKPuTzXH/99ePntt5663Xq33BP22qrrUKZETRsbm6OZdtyyy1Dnz59YtvEeUEAkna/TCaddNL4mdJ/YBBDkrqLwSJJGoPWXHPNsNhii2V/P+CAA2IQgswEOt/PPfdc6NevX/weHVa+xqRvvvkmjrKPO+64oSdV4YH6gw8+CIMGDer0+6effvo2D0k8nPHgccopp4zRYNF4440XqqKsx0qw4cMPPwy/+c1v6n5/4403jg/cyQYbbBDmm2++cNVVV2XBopRB2Ci6oz0aWQR9yBTKBxzXWGONsPzyy8eg0ZFHHpm9ThYGD9k8YJMVBALXXIsEy1dbbbUR/r5FFlmkzTX9+9//PmajXHbZZZUJFo0urgmCcwTmaM+q7v33348BNM6Dc845p833Tj311NgOSJL+T/mG9ySpwa200krhD3/4QxydvuSSSzpcI+T222+PD0eMzpPZMNdcc2Wp6DwEDR48OP55u+22y6ZLpCkHrEnEA+2jjz4asyAIEqV/W1yzKGF6Be9hRJmpVAS0yKTozBo5+Z85omOrt2bR119/HTNzZpxxxhhUoKyMaNdqtTbv4+fstttu4brrrovl473zzjtvuOWWWzodBNphhx3C1FNPHR/uF1xwwfDXv/51uCkTr732WrjpppuyYx/RGi9FU045ZZh77rnDK6+80uZ1slN4KOGY+f0cxy677BI+/fTT7D0EE9t7MFtyySXbBCDr1QfTf8hgS5/l7LPPHo477rg22Vw8CG+00UZt/t38888fy/rUU09lr11xxRXxNQKb3bG+Ep8DU/hmmGGGONoPsrPI8KIclIdy7bfffvH1zl4vHeFc4thmm222TpUjZbDkgyn11iyirBzH22+/HQNM/JnzYp999mkzlSk/jY8HWI6DcnINtbS0DPf7n3/++RjAYgoN5xDnQ72pQs8++2xsbwhI83kSTOlsRl+99qgz1x7/bt99941/JquyeP389NNP4Y9//GNWRj536qhYl/XQjhUz03iNzyF/fn7xxRfxXCDQkwJFYJoRdXDllVeGUUE5uF6LQbT21qJp79okg43vUX7qheP66KOP2v29fDa0CZNMMknM2upsO8Lv4By4++67s3pIbTTZdGR/kaHFv5988snjtcPnlrfKKqvE/19//fVhTCMgRWCK+uR+RYYPbXAR907uTdyjWGeMz/PWW2+N5aP97gjtOveUpZdeerjv8e/5eSNTXz/88EM45JBDwqKLLhrrh2Nadtllw1133dWpMtM2MLWS+kvX0/nnn9+pfytJY1q5howkaSzB+jc8IDHC3V7WCZ18HhBI62c6Gx3Jl19+Odx3333x+0yv4HU6qjvvvHPsoGKppZbKfsbHH38cs5tYh4UHJzqkHTnqqKNih5npHgRVeBjhYYEpHSkDqjM6c2x5dN7p/NPBJpBDtgadfx466UyTnZP3n//8J/z973+PI/2sN8E6UL/+9a/DG2+8ER962sM0IR6W+Bx56OVhluwQHuh4KNhjjz3isbM+BA8IPBikqWU85I8MHoqZ+kJ2Qx4PdAQUCKIxHYKHF7IiHn/88Vi3ZF1tuumm8YGEQEEKuqWHJNbU6CirgewxMi343PhdM800U3zAJKuN6XHUKaiTv/3tb9m/Y3oP5xwP4/fee282RZI/U3Y+lzGNB7BVV101HgsPuAQUeCjm3KDOOZc4jqeffjqeEy+++GIMXHTmeukInw/Bs/ZwPOBY+FwJdvCA3V4mUh5BIdYqIyOFYNC//vWvcNJJJ8WysR5WHhkrrGdGvXEdHn/88TGgx0N0ysajnDzoksnGNDEeTgl+EIy65ppr4poxYNrriiuuGM/D9D4CUSNzHdczomuP46VeOLeoo5SRla4f1hgiOEuwi2vroYceCsccc0wM9lx77bUjfTysQcRXPvOL84Ny54OqIKOStoVrrTO4llJQgAAU05QIjHEtjQqOk+uOshIg4Jzj5xPoo63IlyHfZjFdikXgOXdSe9CZdoRrnWlzBMgOOuig+O/SPYDgFp879UFwlvLxOx577LF4DSYEQDhX+Zm0iWMy44f7A5855eFc4jzh2meto3ReM6hAAJS2jPaawC3XTWeDM0wXBe0+gSmCUqNTX3xuTDlmyiL3cq7f8847L17z+Wmq7ZWZgFgKwnKNcI5xD+TnlnVTBUljkZokqctdcMEFpMPUWlpa2n3PJJNMUlt44YWzvx966KHx3ySnnHJK/PuHH37Y7s/g5/Mefl/R8ssvH7931lln1f0eX8ldd90V3zv99NPXvvjii+z1K6+8Mr7+pz/9KXtt5plnrm2zzTYj/JkdHRv/np+TXHfddfG9Rx55ZJv3bbzxxrVevXrVXn755ew13jfuuOO2ee3JJ5+Mr59++um1jpx66qnxfZdcckn22g8//FBbcskla/37929Tdo5v7bXX7vDn5d+72mqrxbri6+mnn6799re/jb+rqakpe9+9994bX7v00kvb/Ptbbrmlzeuff/55bbzxxqvtvffebd53/PHHx8/jv//9b7v18cc//rE24YQT1l588cU2/3bo0KG1ccYZp/bGG2/Ev1911VXxd7a2tsa//+Mf/4i/c7311qttuumm2b9bYIEFahtuuGGH5X/ttdfizzrhhBNG+DnljzV/nbz77ru1eeedtzbrrLPWXn/99ew9F198ca13797xs8vjvObf3nfffZ2+Xur58ccf42da/Kzz12Txa8CAAbHO6n0G+fOdsvLaEUcc0ea9XPeLLrrocP928sknr33yySfZ69dff318/YYbbsheW3nllWvzzz9/7bvvvsteGzZsWG2ppZaqzTHHHNlre+65Z/y3Dz30UPbaBx98ENsdXud3dqTYHo3Mtcd5UO93PPHEE/H1HXfcsc3r++yzT3z9zjvvrI0sznf+7R133JG9ls7te+65Z7j3b7LJJrVpppmmw5+Z6qPe1+9+97v4eefxOp/XiM73Qw45JL7373//+3DvTT8ztcWU4csvv4xt6hRTTFF7/PHHR7odAddUvl1OFlxwwU63b7Rt88wzzwjf197nkC9Te9L5mr/OKf8ss8xSGzhwYO3nn3+Or5100knxfdwzkm+//bY299xzx9f5XSOy9dZbx/dOOumksW078cQTa88999xw7+tMff3000+177//vs33Pv3009rUU09d23777Tv8fHbYYYfatNNOW/voo4/avG+zzTaL1+k333wz0p+jJHUlp6FJUg9htLejXdHSwrCk/4/qYtBkVzDy3Flks5AtkDD6z+5PLOw6JvHzWViYEeU8Mg/oYxcXpCXbKT9liGwSppuQgTGi38NINKPACSPwaZcksllGFVlijAzzxXQuspP47PNZQIxmM1LPyD0j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csuFFVdcMSy//PJhqaWWimsVSZIkSZIkaSydhrbaaquFBx54IKy33noxy2iZZZYJBx98cLj99tvDN9980zVHKUmSJEmSpGpkFhEYwk8//RRaWlrC3XffHf7973+H448/PvTu3Tt89913XXGckiRJkiRJqkKwKHn11VfD008/HZ588snw1FNPhYkmmihOUZMkSZIkSdJYFCzaYostYjbR999/H4NDrF00dOjQsMACC4RevXp1zVFKkiRJkiSpGsGiyy+/PEwxxRRhxx13DCuttFJcs2iCCSbomqOTJEmSJElStRa4/vjjj8O5554bfvjhh3DAAQfEwBG7oh144IHhtttu65qjlCRJkiRJUjWCRZNOOmncCe3kk08Ojz76aFyvaM455wwnnHBCWHPNNbvmKCVJkiRJklSNaWhkFqUd0PhqbW0NAwYMCOuuu25cv0iSJEmSJEljUbBoqqmmilPPll122bDTTjuFFVZYIcw///xdc3SSJEmSJEmqVrCIaWfzzjtvKJOBAweGiSeeOPTu3TtOk7vrrrt6+pAkSZIkSZLGjmARgaKffvopTkF75ZVXwhZbbBEmmmii8M4778SATf/+/UNPuP/++3vsd6saBg69qdt+1+vHrt1tv0uSJEmSpB4NFv33v/8Na6yxRnjjjTfC999/H1ZdddUYLDruuOPi388666zR/RWSJEmSJEmqym5oe+yxR1hsscXCp59+Gvr165e9vuGGG4Y77rhjpH/ePffcExfHnm666UKvXr3CddddN9x7mpub41Sz8ccfPyyxxBLh4YcfbvN9/h2Law8ePDhceumlo1gySZIkSZKksc9oB4vuvffecPDBB4dxxx23zesEc95+++2R/nlff/11WHDBBWNAqJ4rrrgiDBkyJBx66KHhsccei+9dffXVwwcffJC95z//+U949NFHwz/+8Y9w9NFHx3WVJEmSJEmS1A3T0IYNGxZ+/vnn4V5/66234nS0kbXmmmvGr/acfPLJcde17bbbLv6daW433XRTOP/888PQoUPja9NPP338/7TTThvWWmutGFRaYIEF6v48psrxlXzxxRfx/6zDxBdYKJsvyspXkl6n/LVaLfTtXYuv/zwshGGhV+jTqxZ69frf7/ppWAi10Ct7X9vXQ+hbCN39OCwE/nmf4V7vFX9f/nMnm2qcccYZ7hjbe72zZQLH2x1l6hVq2Wc+pssUXw+1bikTr6dyjfEytfM6P5vfkf980+soXsPtvd6nT59uPfcsk2WyTJbJMlkmy2SZLJNlskyWyTL1HmNlGmPBotVWWy2ceuqp4ZxzzskK+NVXX8XMHwI1XemHH36IGUMHHHBA9hofwiqrrBIeeOCBLDOJD5BAFcdx5513ht/85jft/sxjjjkmHH744cO9/vjjj4cJJ5ww/nnKKacMs802W3jttdfChx9+mL1nhhlmiF8vvvhi+Pzzz8O2c/xfxd3zXq/wwue9woYDh4UBuYSrm9/qHd76OoQtZxvWJuBw9Wu9w1c/hezfJxe+1Dv07xPCxrMMaxOYuPClceLve/7557PXmQJIltVHH30UXn311ez1SSaZJMwzzzxxwXECeElnywSOqzvKNP2EITzyyCPdUibMMUmtW8q05gzDsnKN6TLNOuusYaqppgrPPPNM+Pbbb7PX55577jBgwIB4XucbKoKoZAXmP3cwtZTrLZ+VR+PC1M7uPPcsk2WyTJbJMlkmy2SZLJNlskyWyTLN1mVlevLJJ0Nn9KrlQ02jgEIwDYwf89JLL8UPif9PMcUUcf0hDnJUEXi69tprwwYbbBD/zodG1hA7nS255JLZ+/bbb79w9913h4ceeih+2KyXBCqVLCTWVWpPvcyiGWecMXz88cdxN7eRieTNc8gt//d7uyEL57Vj1uq26CTl6q7MopeOXKNbyoQ5D76l2zKLnjtijVJGkRsxMm6ZLJNlskyWyTJZJstkmSyTZbJMlunnumX65JNPwuSTTx4DTCnmMUYyi4hkEZm6/PLLYxSNbJ4ddtghbLnllm0WvO4uRNE6GynDeOONF7+KqFS+8tKHXpRODIIDeT/Vev1fdKGg+L7/vT78a7V2XqeSi8fX0TGO7OupTMXjHZNlIkDTXWUCgaLuKBOvd/ZcGt0yjej1ep/vyL7enedeZ163TJbJMlmmjl63TJbJMlmmjl63TJbJMlmmjl7vMxaVabj3depdI/ohffqErbbaKoxpZCtR4Pfff7/N6/x9mmmmGeO/X5IkSZIkqdGNUrCIXcZYhLpv377xzx1Zb731QldhbuCiiy4a7rjjjmxqGmlY/H233Xbrst8jSZIkSZI0thqlYBGBmvfeey+uR5SCNvWQblWcszciTGN7+eWXs7+ziNMTTzwRJptssjDTTDOFIUOGhG222SaujbT44ovHxbVZ1DrtjiZJkiRJkqRuDhblF1XK/7krsKr4iiuumP2d4BAIEF144YVh0003jSuAH3LIITFgtdBCC4VbbrklTD311F16HJIkSZIkSWOj0V6z6M0334y7h3WVFVZYoc2K3fUw5cxpZ5IkSZIkSV1v+KW0R9LAgQPD8ssvH/7yl7+ETz/9NFRVc3NzGDRoUBg8eHBPH4okSZIkSVJ1g0VMG2PtoCOOOCJMO+20cQ2jq6++Onz//fehSpqamkJra2toaWnp6UORJEmSJEmqbrBo4YUXDieccEJ44403ws033xymnHLKsPPOO8c1hLbffvuuOUpJkiRJkiRVI1iU3/mMhamZjvavf/0rzDLLLOGvf/1rV/14SZIkSZIkVSlY9NZbb4Xjjz8+7k7GtLT+/fvHdYAkSZIkSZI0Fu2GdvbZZ4fLLrss3HfffWHuuecOW265Zbj++uvDzDPP3DVHKEmSJEmSpOoEi4488siw+eabh9NOOy0suOCCXXNUkiRJkiRJqmawiIWtWa9IkiRJkiRJ1TfaaxYRKLr33nvDVlttFZZccsnw9ttvx9cvvvji8J///KcrjlGSJEmSJElVCRZdc801YfXVVw/9+vULjz/+ePj+++/j659//nk4+uijQ1WwGPegQYPC4MGDe/pQJEmSJEmSqhssYs2is846K/zlL38Jffv2zV5feumlw2OPPRaqoqmpKbS2toaWlpaePhRJkiRJkqTqBoteeOGFsNxyyw33+iSTTBI+++yz0f3xkiRJkiRJqlKwaJpppgkvv/zycK+zXtGss846uj9ekiRJkiRJVQoW7bTTTmGPPfYIDz30UFzs+p133gmXXnpp2GeffcLvfve7rjlKSZIkSZIkdYs+o/sDhg4dGoYNGxZWXnnl8M0338QpaeONN14MFv2///f/uuYoJUmSJEmSVI1gEdlEBx10UNh3333jdLSvvvoq7irWv3//8O2338Zd0iRJkiRJkjSWTENLxh133BgkWnzxxeOuaCeffHKYZZZZuurHS5IkSZIkqczBou+//z4ccMABYbHFFgtLLbVUuO666+LrF1xwQQwSnXLKKWGvvfbqymOVJEmSJElSWaehHXLIIeHss88Oq6yySrj//vvDJptsErbbbrvw4IMPxqwi/j7OOON07dFKkiRJkiSpnMGiq666Klx00UVhvfXWC88880xYYIEFwk8//RSefPLJuI5R1TQ3N8evn3/+uacPRZIkSZIkqXrT0N56662w6KKLxj/PN998cQc0pp1VMVCEpqam0NraGlpaWnr6UCRJkiRJkqoXLCIDh0Wtkz59+sQd0CRJkiRJkjQWTkOr1Wph2223jRlF+O6778Kuu+4aJpxwwjbv+/vf/z76RylJkiRJkqRyB4u22WabNn/faqutuuJ4JEmSJEmSVMVg0QUXXNC1RyJJkiRJkqTqrlkkSZIkSZKkxmOwSJIkSZIkSRmDRZIkSZIkScoYLJIkSZIkSdLoLXC9yCKLhDvuuCNMOumk4Ygjjgj77LNPmGCCCUblR0nqYgOH3tRtv+v1Y9futt8lSZIkSSpxZtFzzz0Xvv766/jnww8/PHz11Veh6pqbm8OgQYPC4MGDe/pQJEmSJEmSqpVZtNBCC4XtttsuLLPMMqFWq4UTTzwx9O/fv+57DznkkFAFTU1N8euLL74Ik0wySU8fjiRJkiRJUnWCRRdeeGE49NBDw4033hh69eoVbr755tCnz/A/iu9VJVgkSZIkSZKkUQwWzTXXXOHyyy+Pf+7du3dcv2iqqabq6mOTJEmSJElSFYJFecOGDeuaI5EkSZIkSVL1g0V45ZVXwqmnnhoXvgYLRe+xxx5httlm64ofL0mSJEmSpDLvhpZ36623xuDQww8/HBZYYIH49dBDD4V555033H777V1zlJIkSZIkSapGZtHQoUPDXnvtFY499tjhXt9///3DqquuOrq/QpIkSZIkSVXJLGLq2Q477DDc69tvv31obW0d3R8vSZIkSZKkKgWLppxyyvDEE08M9zqvuUOaJEmSJEnSWDYNbaeddgo777xzePXVV8NSSy0VX7vvvvvCcccdF4YMGdIVxyhJkiRJkqSqBIv+8Ic/hIkmmiicdNJJ4YADDoivTTfddOGwww4Lu+++e1ccoyRJkiRJkqoSLOrVq1dc4JqvL7/8Mr5G8EiSJEmSJEljYbAozyCRJEmSJEnSWL7AdaNobm4OgwYNCoMHD+7pQ5EkSZIkSeoxBot+0dTUFFpbW0NLS0tPH4okSZIkSVKPMVgkSZIkSZKkrgkW/fjjj2HllVcOL7300uj8GEmSJEmSJDVCsKhv377hqaee6rqjkSRJkiRJUrWnoW211VbhvPPO65qjkSRJkiRJUo/qM7o/4Keffgrnn39++Ne//hUWXXTRMOGEE7b5/sknnzy6v0KSJEmSJElVCRY988wzYZFFFol/fvHFF9t8r1evXqP74yVJkiRJklSlYNFdd93VNUciSZIkSZKk6q9ZlLz88svh1ltvDd9++238e61W66ofLUmSJEmSpKoEiz7++OOw8sorhznnnDOstdZa4d13342v77DDDmHvvffuimOUJEmSJElSVYJFe+21V+jbt2944403wgQTTJC9vummm4ZbbrlldH+8JEmSJEmSqrRm0W233Rann80wwwxtXp9jjjnCf//739H98ZIkSZIkSapSZtHXX3/dJqMo+eSTT8J44403uj9ekiRJkiRJVQoWLbvssuGiiy7K/t6rV68wbNiwcPzxx4cVV1xxdH+8JEmSJEmSqjQNjaAQC1w/8sgj4Ycffgj77bdfePbZZ2Nm0X333dc1RylJkiRJkqRqZBbNN9984cUXXwzLLLNMWH/99eO0tI022ig8/vjjYbbZZuuao5QkSZIkSVI1MoswySSThIMOOihUWXNzc/z6+eefe/pQJEmSJEmSqh0s+vTTT8N5550Xnnvuufj3QYMGhe222y5MNtlkoSqampri1xdffBGDX5LKZ+DQm7rtd71+7Nrd9rskSZIkqaGmod1zzz1h4MCB4bTTTotBI7748yyzzBK/J0mSJEmSpLEos4hsnE033TT8+c9/DuOMM058jalcv//97+P3nn766a44TkmSJEmSJFUhs+jll18Oe++9dxYoAn8eMmRI/J4kSZIkSZLGomDRIosskq1VlMdrCy644Oj+eEmSJEmSJJV9GtpTTz2V/Xn33XcPe+yxR8wi+tWvfhVfe/DBB+POYscee2zXHakkSZIkSZLKGSxaaKGFQq9evUKtVste22+//YZ73xZbbBHXM5IkSZIkSVIDB4tee+21rj8SSZIkSZIkVTNYNPPMM3f9kUiSJEmSJKnHjVKwqOidd94J//nPf8IHH3wQhg0b1uZ7rGkkSZIkSZKksSRYdOGFF4ZddtkljDvuuGHyySePaxkl/NlgkSRJkiRJ0lgULPrDH/4QDjnkkHDAAQeE3r17d81RSZIkSZIkqUeMdnTnm2++CZtttpmBIkmSJEmSpAYw2hGeHXbYIVx11VVdczSSJEmSJEmq9jS0Y445JqyzzjrhlltuCfPPP3/o27dvm++ffPLJo/srJEmSJEmSVKVg0a233hrmmmuu+PfiAteSpI4NHHpTt/2u149du9t+lyRJkqSxNFh00kknhfPPPz9su+22XXNEkiRJkiRJqu6aReONN15Yeumlu+ZoJEmSJEmSVO1g0R577BFOP/30rjkaSZIkSZIkVXsa2sMPPxzuvPPOcOONN4Z55513uAWu//73v4/ur5AkSZIkSVJVgkUDBgwIG220UdccjSRJkiRJkqodLLrgggtCI2hubo5fP//8c08fiiRJkiRJUnXXLGoUTU1NobW1NbS0tPT0oUiSJEmSJFU3s2iWWWYJvXr1avf7r7766uj+CkmSJEmSJFUlWLTnnnu2+fuPP/4YHn/88XDLLbeEfffdd3R/vCRJkiRJkqoULNpjjz3qvs76P4888sjo/nhJkiRJkiQ1wppFa665ZrjmmmvG1I+XJEmSJElSlYJFV199dZhsssnG1I+XJEmSJElSGaehLbzwwm0WuK7VauG9994LH374YTjzzDNH98dLkiRJkiSpSsGiDTbYoM3fe/fuHaaccsqwwgorhLnnnnt0f7wkSZIkSZKqFCw69NBDu+ZIJEmSJEmS1LhrFkmSJEmSJGksyixiull+raJ6+P5PP/00qr9CkiRJkiRJVQkWXXvtte1+74EHHginnXZaGDZs2Kj+eEmSJEmSJFUpWLT++usP99oLL7wQhg4dGm644Yaw5ZZbhiOOOGJ0j0+SJEmSJElVW7PonXfeCTvttFOYf/7547SzJ554Ivz1r38NM888c1f8eEmSJEmSJFUhWPT555+H/fffP8w+++zh2WefDXfccUfMKppvvvm67gglSZIkSZJU/mloxx9/fDjuuOPCNNNME/72t7/VnZYmSZIkSZKksSRYxNpE/fr1i1lFTDnjq56///3vo3N8kqSKGjj0pm77Xa8fu3a3/S5JkiSp0Y1ysGjrrbcOvXr16tqjkSRJkiRJUjWDRRdeeGHXHokkSZIkSZIaYzc0SZIkSZIkNQaDRZIkSZIkScoYLJIkSZIkSVLGYJEkSZIkSZIyBoskSZIkSZKUMVgkSZIkSZKkjMEiSZIkSZIkZQwWSZIkSZIkKWOwSJIkSZIkSRmDRZIkSZIkScoYLPpFc3NzGDRoUBg8eHBPH4okSZIkSVKPMVj0i6amptDa2hpaWlp6+lAkSZIkSZJ6jMEiSZIkSZIkZQwWSZIkSZIkKdPnf3+UJEkjMnDoTd32u14/du1u+12SJElSYrBIkiRFBsIkSZIEp6FJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpTp878/SpIkqSoGDr2p237X68eu3W2/S5Ik9TwziyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpQxWCRJkiRJkqTGDxZ98803YeaZZw777LNPTx+KJEmSJElSZTRssOioo44Kv/rVr3r6MCRJkiRJkiqlIYNFL730Unj++efDmmuu2dOHIkmSJEmSVCmlCxbdc889Yd111w3TTTdd6NWrV7juuuuGe09zc3MYOHBgGH/88cMSSywRHn744TbfZ+rZMccc041HLUmSJEmS1BhKFyz6+uuvw4ILLhgDQvVcccUVYciQIeHQQw8Njz32WHzv6quvHj744IP4/euvvz7MOeec8UuSJEmSJEkjp08oGaaOdTR97OSTTw477bRT2G677eLfzzrrrHDTTTeF888/PwwdOjQ8+OCD4fLLLw9XXXVV+Oqrr8KPP/4YJp544nDIIYfU/Xnff/99/Eq++OKL+P+ffvopfqF3797xa9iwYfErSa///PPPoVarhb69a/H1n4eFMCz0Cn161UKvXv/7XT8NC6EWemXva/t6CH0Lobsfh4XAP+8z3Ou94u/j9yZkYY0zzjjDHWN7r3e2TOB4u6NMvUIt+8zHdJni66HWLWXi9VSuMV6m3v93EN1Rpvjvf/qpW8qUXs8f/5gqE9Kv744y5duOMV2mn2q9uq1MoBzdUabenWw7uqJM8eeHWreUaZxOtB1dVSZ+Nr+jO8qUcAzdUab8uZFeT7+/M6/36dOnW++5I1Omtm3HmKmn9toO68kyWSbLZJksk2VqjDJVJljUkR9++CE8+uij4YADDshe40NYZZVVwgMPPBD/zvSzNAXtwgsvDM8880y7gaL0/sMPP3y41x9//PEw4YQTxj9POeWUYbbZZguvvfZa+PDDD7P3zDDDDPHrxRdfDJ9//nnYdo7/q7h73usVXvi8V9hw4LAwYNz//cyb3+od3vo6hC1nG9amQ3f1a73DVz+F7N8nF77UO/TvE8LGswxr0/G78KVx4u9jXaakX79+Mcvqo48+Cq+++mr2+iSTTBLmmWee8M4774S33nore72zZQLH1R1lmn7CEB555JFuKRPmmKTWLWVac4ZhWbnGdJlmnXXW+P/uKBMoV3eUaaqpporXcv44x1SZ8NkP//f/7ijTt99+mx3rmC7TVa+N021lAsfbHWXiWs63HWOyTOA4uqNMy03zv3KN6TLNPffcYcCAAd1SpoRj644ycT/Pd9AWWGCBMO6447Y5Z7DYYovFfsZTTz3VplM1ePDgbr3njkyZUr2MyXp665sQbn5zHOvJMlkmy2SZLJNlCo1RpieffDJ0Rq9aPtRUMkS9rr322rDBBhvEv/OhTT/99OH+++8PSy65ZPa+/fbbL9x9993hoYceavPvU7DoxBNPbPd31MssmnHGGcPHH38cM5JGJpI3zyG3xNe7IwvntWPW6rboJOXqrsyil45co1vKhDkPvqXbMoueO2KNbosiz3rgzd2WWUS5ujMyPvcfbh7jZQK//uVj1um2aH9qO7ojC+fVo9fsthEMytVdmUUvdqLt6KpRmTkOvqXbMotG1HZ09UjTHAfeOMbLlDz/xzUdERzNMrVtO8ZsZlGx7bCeLJNlskyWyTJZpmqW6ZNPPgmTTz55DDClmEflM4tG1rbbbjvC94w33njxq4hK5SsvfehF6cSg85VH5yr23gqK7/vf68O/VmvndSq5eHwdHePIvp7KVDzeMVkmHr66q0wgUNQdZeL1zp5Lo1umpDvKhHy5xnSZeL3e8Xd1mUbn2EelTPXK0AhlKpZjTJZpWBe1HZ0pUzqO7ijTsC5oOzpbpv8d+5gvU/EYxnSZ6p0bI/t6d95zO/N6Opbh246ur6cRHbv1ZJlG9nXLZJk6OnbLZJksU+ixMg33+0KFTDHFFLHA77//fpvX+fs000zTY8clSZIkSZLUKCoVLGJu4KKLLhruuOOO7DXSsPh7flqaJEmSJEmSRk3ppqGxg9nLL7+c/Z1FnJ544okw2WSThZlmmikMGTIkbLPNNnEBqcUXXzyceuqp4euvv852R5MkSZIkSVIDBYtYVXzFFVfM/k5wCASIWLB60003jSuAs8PZe++9FxZaaKFwyy23hKmnnroHj1qSJEmSJKkxlC5YtMIKK7RZsbue3XbbLX5JkiRJkiSpwYNFPaW5uTl+Fbe6kyRJ1TZw6E3d9rteP3btbvtdkiRJY0qlFrgek5qamkJra2toaWnp6UORJEmSJEnqMQaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmDRL5qbm8OgQYPC4MGDe/pQJEmSJEmSeozBol80NTWF1tbW0NLS0tOHIkmSJEmS1GMMFkmSJEmSJCnT539/lCRJkqSRM3DoTd32u14/du1u+12SNDYzs0iSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBol80NzeHQYMGhcGDB/f0oUiSJEmSJPUYg0W/aGpqCq2traGlpaWnD0WSJEmSJKnHGCySJEmSJElSxmCRJEmSJEmSMn3+90dJkiRJY8rAoTd12+96/di1u+13SZIar603s0iSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFv2iubk5DBo0KAwePLinD0WSJEmSJKnHGCz6RVNTU2htbQ0tLS09fSiSJEmSJEk9xmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBol80NzeHQYMGhcGDB/f0oUiSJEmSJPUYg0W/aGpqCq2traGlpaWnD0WSJEmSJKnHGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0W/aG5uDoMGDQqDBw/u6UORJEmSJEnqMQaLftHU1BRaW1tDS0tLTx+KJEmSJElSjzFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0W/aG5uDoMGDQqDBw/u6UORJEmSJEnqMQaLftHU1BRaW1tDS0tLTx+KJEmSJElSjzFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnK9PnfHyVJkiRJGDj0pm77Xa8fu3a3/S5J6gwziyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyhgskiRJkiRJUsZgkSRJkiRJkjIGiyRJkiRJkpQxWCRJkiRJkqSMwSJJkiRJkiRlDBZJkiRJkiQpY7BIkiRJkiRJGYNFkiRJkiRJyvT53x8lSZIkSY1s4NCbuu13vX7s2t32uyR1LTOLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLftHc3BwGDRoUBg8e3NOHIkmSJEmS1GMMFv2iqakptLa2hpaWlp4+FEmSJEmSpB5jsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIkmSJEmSJGUMFkmSJEmSJCljsEiSJEmSJEkZg0WSJEmSJEnKGCySJEmSJElSxmCRJEmSJEmSMgaLJEmSJEmSlOnzvz8KtVot/v+LL74Y6X877PtvQncZleMbVZZr9FmurtGoZbNco89ydY1GLZvlqt652Kgatc4s1+izXFLjGFbyayz9mxT7aE+v2ojeMZZ56623wowzztjThyFJkiRJkjRGvPnmm2GGGWZo9/sGiwqGDRsW3nnnnTDRRBOFXr16jdHfRUSPwBSVNPHEE4dG0ajlauSyWa5qadRyNXLZLFe1NGq5GrlslqtaGrVcjVw2y1UtjVquRi7bF91YLkJAX375ZZhuuulC797tr0zkNLQCPqyOomtjAidDI53ojV6uRi6b5aqWRi1XI5fNclVLo5arkctmuaqlUcvVyGWzXNXSqOVq5LJN3E3lmmSSSUb4Hhe4liRJkiRJUsZgkSRJkiRJkjIGi3rQeOONFw499ND4/0bSqOVq5LJZrmpp1HI1ctksV7U0arkauWyWq1oatVyNXDbLVS2NWq5GLtt4JSyXC1xLkiRJkiQpY2aRJEmSJEmSMgaLJEmSJEmSlDFYJEmSJEmSpIzBIo31vvjii54+BEmSJEmSSsNgURd7/fXXQ6P66KOP4v8baU30iy66KAwcODA8/PDDodE0Yn3hjTfeCD///HNoNPfee2/44YcfQqO59dZb43XWiBq1bI1aruuuuy6ccMIJodFcdtllYd999w2NqFHPRdr7W265JTSaRi0XnnvuufD222+HRvPoo4+Gxx9/PDQa245qadT6aoiysRuausaFF15YW2CBBWpnnXVWrdGce+65tV69etVuuumm+Pdhw4bVqu68886LZVp44YVr0003Xe2hhx6qNYpGrC+cc845sVxnnnlmrZGkci2yyCK17777rtYoTj/99Fiu+eefP7aPjaRRy9ao5frTn/4UyzX11FPXjj766FqjOOWUU2K5+Pp//+//1RpJo56LZ5xxRizXyiuvXLv11ltrjaJRy4U///nPsWx77rln7cMPP6w1CvpSlGuLLbaoPfXUU7VGYdtRLY1aX41SNjOLujBquN1228U/X3HFFeG8884LjeKqq64KO+20U1hmmWXCJptsEv75z3+GXr16VT5jZdFFFw033nhjuOGGG2LZ1l133YbIMGrU+mIkZZ999gkbbLBB2GuvvcJZZ50Vqo46YbRyl112CQceeGAYd9xxwworrBC+//77UHUff/xxvLaOPfbYsOSSS4ZzzjknXHDBBaERNGrZGrVcZOzdc8894bTTTgt77LFH+Otf/xqOOuqo0AieeuqpcO6558bsossvvzzstttuoRE06rlI9vkll1wSDjnkkNC7d+9w+umnx/5j1TVqufDAAw+EY445JjQ1NYXm5ub45w8//DBU3TPPPBPLQ7+qpaUlnHTSSbE9qTrbjmpp1PpqqLL1dLSqUdx333215557LmYF/PrXv64ts8wyMbujEVCuO+64I2an/P73v6/17du3shkr6Xh//vnnNq8/++yztd/85je1qaaaqvIZRo1UX3mvvfZa7YEHHoh/JjOgd+/ecbSvEbz11lvx/7fffnttoYUWqv3qV7+qfIYR59qnn34a//zMM8/Ufvvb39aWWmqpmNFXdY1atkYtF3744Yf4/zfeeKN28MEH1+aee+7akUceWauq4j3ss88+i32OKaecMrb7Vdeo5+K3335be/vtt+Of77///tpyyy1XW2eddWo333xzrcoatVwgk+jJJ5+Mf77qqqtilsBee+1V++CDD2pV9vnnn9deeuml+GeyVGaeeeba1ltvnZW1qmw7qqVR66uRymawaDS09+D93nvvNUzAqNgh5aRvamqqZADip59+avPQUNQIAaNGqq/8sdY75kYIGKX6+vHHH7Nz9F//+ldlA0apPPXqrbW1tbI3ymJZ+HP6qnrZGrXO8uWgjHyl1958883aH/7wh0oGjNJ9rF6bz8MffQ7uYVUMGI1N11j684MPPljph77ivTmVtVHKVWxHcPXVVzdMwChfrttuu62yAaNGvY816jU2ttTXsAa5jxks6oLAAyN6xQuAgNHGG29cyYBRMeBQVMUARKovOtNcpO3N9a1iwKgR6ytfZ4ymtOeYY46pXMBoRPVV1YBRqq8vv/wyzsv+5ptvhntPVW+UqWyp7lLAOV+XVSxbo9ZZKldHbQjZfFULGOXvY7vuumvto48+Gu49X3zxRSUzjBr9Gvvqq69iu54GBlDlgFHxYS8/8FHlcuXrjPaQekvyD31VDBi11/egzqocMGrU+1ijXmONWl+NfB8zWDQaJwNBokUXXTR2APKqHDDK3yT//e9/x5RbUnDTxZwaqCoFIPId7Omnn7620UYbZd9Lx5zvwFUpYNSI9VWsswknnLB29tlnN0TAKLUN3CSPOOKI2pAhQ2onnXRS7eWXX650wChfX5NPPnmbhXaLI0dVu1GmOuMhfLvttqutt956sWN2yy23tGk3qla2Rq2zfH2R4bvSSivFjSe4B6fpnvUCRn/84x9rVakvNmRgQdp6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0LMQMmvclLnv0aj3sUa7xuwrVu8+1lm9+E9PZzeVRUqV+/LLL8NOO+0UXn/99fja+uuvH/bcc88w4YQTDvfe5Omnnw733HNPmHfeecMKK6wQ09TKkhKYTwsknZHpO8suu2y4/PLLO0zJbS/Vrixly5dpvvnmC0suuWQ44IADwuabbx5TAKnDYjokHn/88TDTTDOFySeffIRl7SnpeCjbeuutF///4osvhi233DJsuummccoZOH5Uob6K1xhleffdd8Ozzz4bdtttt3DYYYe1mUZWPC+py7PPPjumrpLaWaZyFc9FjvGGG25o855Unny5mGpS5jTvfLlo2ygX0yZ4/brrrhvhv3///ffDt99+GwYOHFiq+srjXFxooYXilINPP/00fPXVV2GqqaaKqcOUt4ztQ2evsWWWWSamP7/wwgtxesFaa60VjjzyyErWWf5cJI2br3/84x/tvi/5+uuv29y7y6bYdjBdiT7EeeedFy677LIwaNCguucf01WZVgem1k055ZSlqq+x5RqjrpiC9Oijj4Zf/epX8b62yy67ZO+tV65nnnkm/ps555yzVHWWPxepL849pifVKz/HWzzmMtdh8V7GVLInn3wynHjiiW3qq71yXHvttbHfuOiii5aqztKxci4yBZU+FOcXZaSNZJmFjuqsvZ/X08aG+1gjXWP2FRes3H1spPR0tKosUpSPFHB2WWI1c7Ye33nnneOCVPV2HkF+Jff8AndlihqmY2EUgUWBN9544zbfLy7End9KmHTpMiOCzRanZEol7DrH6Gy9hfnSNsIsLJzf0rCMWLia9OCNNtoo7nBANg3bJbNwWnGngCrUV/EaY6oF1xUL+JFa294IBGWvwoLPlIvdRChXwpan1Em9XfWYTkc2QNnPQ46dcrHTXH6L8fZ2fmGEaMiQIbUq4JwkW40Fj1O24UsvvRQXJqStJGOgWLbiLlplxEgs9y3qjDaSRRZZ2JS2o7jYfZXqjGuMjAzS8fOLSTLqXC91/Ywzzqhtvvnmw20HXTacd/lrjOwTsmSPPvrouu9nh6LlllsubrJRdo16jZF1zkLB9D3ITiGTg2ww+o71+n/siMZodNnRTyRLm+3hE0bFWaA7n42eRs8pF1PqyizVB23hTDPNlPWBySxnyn4xazthSngZp5wVURdrr712/GK6HAun00ekTVl++eWHy+qm78GS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OH908803D9NNN118/yGHHBLuu+++cPbZZ8d1qcrojTfeCDPPPHN48cUXw8033xyuv/768PDDD4dFF100frEm06OPPhrnx6a6oU5feeWV+DXDDDOEMs/r5RhfffXVrE4GDx4crz3WMTrvvPPCvPPOm83xLXO50loAo2KBBRYI//znP2P7kuZuN8KaVLPNNlt46623Yt2WrVz5+mJ9g5Fds22mmWYKjz/+eJhqqqlKV7ZGrDPrq3rlytdZuhePjPXWWy/es5dYYonSlW1UDRgwIK47uMIKK8TyNdI1tsEGG4Tdd9899rfKVK5GrrPR6YNUpVwjizWMNtpoo3DEEUeE1VZbrWGuMdYnYo1L1hbkWa1R1vSk38taWazJxDNN2co1OvexaaedNmy44Yax3niOK1vZKqdMu2ilqT3sTjHvvPPG6DSZOClbhe8z6soaAfnMB6YKbbvttnHtkbJju89DDjkklo01fhhNTtPLmL/Ma2SvFEf5Ntlkk1rZ7bHHHrWJJ5442/45v4sYa+Nst9128c+pntl1itHZskfed95551r//v3j7hXs6pC2Z2QUj2yq/LQtbLrppjH7qMxeffXV2sILLxzPtTS9LGEHJjLdWHuqzOUa3V0Fyzqts97ox8gca/HzKNNoSqqztIvKyEplKVOZGrnOGrW+6tXNyBxjWcuVr7O0A9PoKksZR+caq7dcQSNcY/XKUJZyjQ11Niq7Npe5XKNTX/XWuGzUa6wsurK+Gv0+ptHTo9twpUghu2gRhX7ggQfi69NPP3245ZZb4g4c22+/fcxUISrILhCMoBAVfv7557OfwwruRA/ZXazspplmmrDzzjvHbJqDDz44RqxT5HS55ZYLs88+e7jjjjtiGfnCaaedFrNYyiod5zHHHBNWXnnlrC7zUWDqiJHY/G4erL6fXiubFIUm8v773/8+/O53vwuHHnpoHP1P5SJqPffcc4cbb7wx/p2MHFx00UXhyCOPDGU2yyyzxJGtww47LNx0001tvrfMMsvEUYa///3vsW65TstWrq7YVbCMuxCmXerY2YadBO+///6RHgVL7Un6N2UY1SvuLEidcX6NrFSWspSpkeuskeuLz/m7776L2a033HBDLGNqvzszAlnGchXrjGxXMkRHVxnKmK4xdll67bXXwgsvvBBf7+w1VnxffvemKl9j9cpQhnKNDXVG32O7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3333dXfeeWf0Hmy84oor3O677+4qVarkTjnlFLdkyZKt+GQNwzAMwyhuzFlkGIZhGMZ2B0fN/Pnz3Y477hh9D8fHo48+6u6//3734Ycfuh49eriLLrrI/fvf//ZOnDZt2ngnRMiECRO8swXHSRycGaeffrqrWLGimzdvnnv99dddhQoVfIQTjheibnB88PuB94RfA/8m1ay4wXFy2mmneScYzhucWdhy0003udtuu81HAQ0cONDdeOON7pFHHvE/c9ddd7nnnnvOPf744+6TTz7x78chUxTn1fTp0/N85uKEmzp1qvvoo4/8/3/99df7/yNk9uzZbvny5f5vnmfcuHH+j9CuXTv31Vdf+bpMTz75pBszZoz75ptv8vwOHFV876WXXvKRZfXq1XOnnnqqpcUZhmEYhgYyhmEYhmEYfzLt27fPlC5dOrPTTjtlypQpk2FLUqpUqcwTTzzhX9+0aVOmfPnymfnz5+f5ucsvvzzTpk0b/+933nkn85e//CWzatUq//Xvv/+e2XvvvTOjRo2K3s/vffrpp/2/x48fnznooIMy//vf/6LXN2/enClXrlxm+vTp/uuWLVtmunbt6v/dvXv3TK9evTJVqlTJLF26NPPLL7/4Z5oxY0ZOuxo1apS5+uqrs762YsUK/zw8N8yePdt/ze+uW7duplWrVv55hP333z8zceLEPL9jwIABmfr16/t/d+vWLXPKKafksSc/+vXr5z9jPvOyZcv6/5s/w4cPz/fn+Dx4trDtatasmfntt9+i751//vmZCy+80P8be/i9b775ZvT6Z5995r83YsQI//W8efMylSpV8u0cgs2jR48ulD2GYRiGYfxxWM0iwzAMwzC2C40bN/ZpZD///LMbMWKET5Fq1aqVf23ZsmU+LYtomxAigI466ij/7yOPPNKnaBFd1KdPHx/1Q6QKESvZIMWJ30tkUQhFtYmSgUaNGvkoGOD3Ec3z6aef+ggZIl6ITiJyqTjBxuOOO87XDaKuEfCZ8EykhHXo0CF672+//eZ23nnnKB2MnyVVjOioZs2auaZNm+b7f/FeopGw+bHHHvMpcd26dcvznnvvvdc99NBD7osvvnAbN270nzmfdQhRWPKsQDra+++/7/9NlBNtSaSQcMABB7gqVarkaYuffvrJR26F8P9JWxiGYRiGsf0wZ5FhGIZhGNuFnXbayTsRAOcENYkefPBB7yDBkQAvvviir68Tr3Uk/OMf/4icRfyN0yTugBD4nUcffbRP14pD3ZxQzeyzzz7zaVjUJ/r444+9s+j77793xxxzjCtfvnyxfg7U9iFVi/+vTp060bPC2LFj3d/+9rc87xcnDc6YFStW+DQuilZTi4k6RFLXKRuknMlnPnjwYP9/33zzzW7AgAH+e5MnT3Y9e/Z0d9xxh6tfv753rA0dOtS98cYbeX4PCnYh1GIiha2wYB8OJj7XOKTgGYZhGIaxfTFnkWEYhmEY2x1qEFEb55prrnFt27Z1hx56qHcKEd1CtE8ueO8NN9zga97gJKG+US5wrhC9U7VqVV9QORs4a4iAoZA10TTUNMKBhEIbzqI/ol4RThv+H+r14DzB9j322MNVq1bNff75594hlgvsuPDCC/2f8847zzvLiICieHdh4LOjsHTnzp39/0cdpxNOOMF16dIlek9RI32IXiICimLjOOeAiC4+v7Atvv76ax+BVJQ6S4ZhGIZh/DlYgWvDMAzDMFRA+hhRM6RBEdFChAtFrSmgjMPi7bffdnfffXdU4BlwNODcIBoJJa9zzjkn5+/H6bLbbrt5BTSKVxOVg3PmqquucmvWrIkiZE466SQffSSOobp167rNmze7V155JV/HlYCaGuld4Z9169bl+zMoi/F8OG6IZAIifijyTSFrUuFI83r44Yfd8OHD/ev8PWnSJP9+XqcoNUpnRYnMIXoI+0i3gwMPPNC99dZbvvA1v5OC2m+++aYrCqi2EeHUsWNHt2jRIu804t8oo/H5Aq/zf6OqNmPGDF9smwLnffv29f+/YRiGYRjbF3MWGYZhGIahAqJMkIIfMmSIr9lDahTOChwm1CYiaoa0NKTYQ3CyUAOnRYsW3iGRC9LH5s6d6/bZZx/XsmVL/ztxMlG/J4w0wiGE40mcRUQ94UDC0VGYekWkw1FXKfxDOllBULeJVDIcRjhqkJV/4IEHvIOIiCeeC8UxsR+HGp8VqXHHHnusd7hMmzbNP29RwCHH/7N69WrXqVMn/9kQqUT62/r16/NEGRUWVOyIjuJzo12ou8Tzli1b1r/OZ8mz8vqll17qateu7Vq3bu1WrVrlf84wDMMwjO3LX6hyvZ2fwTAMwzAMw0gxRG7VqFHD11Yi3c4wDMMwDN2Ys8gwDMMwDMMoVl599VVfxJqIqLVr17revXu7L7/80kdMxYtjG4ZhGIahDytwbRiGYRiGYRQrv/76qy9YToFu0s+oK0UdKHMUGYZhGEYysMgiwzAMwzAMwzAMwzAMI8IKXBuGYRiGYRiGYRiGYRgR5iwyDMMwDMMwDMMwDMMwIsxZZBiGYRiGYRiGYRiGYUSYs8gwDMMwDMMwDMMwDMOIMGeRYRiGYRiGYRiGYRiGEWHOIsMwDMMwDMMwDMMwDCPCnEWGYRiGYRiGYRiGYRhGhDmLDMMwDMMwDMMwDMMwjAhzFhmGYRiGYRiGYRiGYRhO+D/V8skTy0yG/gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1400x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 1: Use pd.cut directly on a copy\n",
    "like_bins = pd.cut(dat['book_review_likes'], bins=20)\n",
    "\n",
    "# Step 2: Get distribution counts\n",
    "like_bin_distribution = like_bins.value_counts().sort_index()\n",
    "\n",
    "# Step 3: Plot\n",
    "plt.figure(figsize=(14, 7))\n",
    "like_bin_distribution.plot(kind='bar')\n",
    "\n",
    "plt.yscale('log')\n",
    "plt.title('Distribution of Review Likes (Binned into 20 Buckets) [Log-Scale]')\n",
    "plt.xlabel('Review Likes Range')\n",
    "plt.ylabel('Number of Reviews')\n",
    "plt.grid(True, axis='y', linestyle='--', alpha=0.7)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>review_text</th>\n",
       "      <th>date_added</th>\n",
       "      <th>n_votes</th>\n",
       "      <th>n_comments</th>\n",
       "      <th>review_likes</th>\n",
       "      <th>text_reviews_count</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>average_rating</th>\n",
       "      <th>book_review_likes</th>\n",
       "      <th>like_share</th>\n",
       "      <th>popular</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>5</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>Sun Jul 30 07:44:10 -0700 2017</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>374</td>\n",
       "      <td>6336</td>\n",
       "      <td>4.01</td>\n",
       "      <td>199</td>\n",
       "      <td>0.145729</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>16981</td>\n",
       "      <td>3</td>\n",
       "      <td>Recommended by Don Katz. Avail for free in Dec...</td>\n",
       "      <td>Mon Dec 05 10:46:44 -0800 2016</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3741</td>\n",
       "      <td>125232</td>\n",
       "      <td>3.84</td>\n",
       "      <td>631</td>\n",
       "      <td>0.001585</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>28684704</td>\n",
       "      <td>3</td>\n",
       "      <td>A fun, fast paced science fiction thriller. I ...</td>\n",
       "      <td>Tue Nov 15 11:29:22 -0800 2016</td>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>22</td>\n",
       "      <td>1026</td>\n",
       "      <td>13990</td>\n",
       "      <td>4.10</td>\n",
       "      <td>1010</td>\n",
       "      <td>0.021782</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>27161156</td>\n",
       "      <td>0</td>\n",
       "      <td>Recommended reading to understand what is goin...</td>\n",
       "      <td>Wed Nov 09 17:37:04 -0800 2016</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>13663</td>\n",
       "      <td>99022</td>\n",
       "      <td>3.96</td>\n",
       "      <td>4340</td>\n",
       "      <td>0.001382</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>32283133</td>\n",
       "      <td>0</td>\n",
       "      <td>http://www.telegraph.co.uk/culture/10...</td>\n",
       "      <td>Tue Nov 01 11:09:18 -0700 2016</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>9</td>\n",
       "      <td>3061</td>\n",
       "      <td>17494</td>\n",
       "      <td>3.86</td>\n",
       "      <td>1436</td>\n",
       "      <td>0.006267</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  18245960       5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d     16981       3   \n",
       "2  8842281e1d1347389f2ab93d60773d4d  28684704       3   \n",
       "3  8842281e1d1347389f2ab93d60773d4d  27161156       0   \n",
       "4  8842281e1d1347389f2ab93d60773d4d  32283133       0   \n",
       "\n",
       "                                         review_text  \\\n",
       "0  This is a special book. It started slow for ab...   \n",
       "1  Recommended by Don Katz. Avail for free in Dec...   \n",
       "2  A fun, fast paced science fiction thriller. I ...   \n",
       "3  Recommended reading to understand what is goin...   \n",
       "4           http://www.telegraph.co.uk/culture/10...   \n",
       "\n",
       "                       date_added  n_votes  n_comments  review_likes  \\\n",
       "0  Sun Jul 30 07:44:10 -0700 2017       28           1            29   \n",
       "1  Mon Dec 05 10:46:44 -0800 2016        1           0             1   \n",
       "2  Tue Nov 15 11:29:22 -0800 2016       22           0            22   \n",
       "3  Wed Nov 09 17:37:04 -0800 2016        5           1             6   \n",
       "4  Tue Nov 01 11:09:18 -0700 2016        9           0             9   \n",
       "\n",
       "  text_reviews_count ratings_count average_rating  book_review_likes  \\\n",
       "0                374          6336           4.01                199   \n",
       "1               3741        125232           3.84                631   \n",
       "2               1026         13990           4.10               1010   \n",
       "3              13663         99022           3.96               4340   \n",
       "4               3061         17494           3.86               1436   \n",
       "\n",
       "   like_share  popular  \n",
       "0    0.145729        1  \n",
       "1    0.001585        0  \n",
       "2    0.021782        1  \n",
       "3    0.001382        0  \n",
       "4    0.006267        0  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# calculate like share\n",
    "dat[\"like_share\"] = dat[\"review_likes\"]/dat[\"book_review_likes\"]\n",
    "\n",
    "# create popularity binary variable\n",
    "popular_thresh = 0.02\n",
    "dat[\"popular\"] = np.where(dat[\"like_share\"]>popular_thresh,1,0)\n",
    "\n",
    "dat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pre english filter popularity count:\n",
      "popular\n",
      "0    1725317\n",
      "1     301903\n",
      "Name: user_id, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"pre english filter popularity count:\")\n",
    "print(dat.groupby(\"popular\")[\"user_id\"].count())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 320900,
     "status": "ok",
     "timestamp": 1745698548784,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "-ijm94jvsRwB",
    "outputId": "c5af5675-ea68-460d-b2fb-07c39ff02ae5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "english reviews:  1710198\n"
     ]
    }
   ],
   "source": [
    "# filter to books in English\n",
    "language_model = fasttext.load_model('lid.176.bin')\n",
    "pred = language_model.predict(dat[\"review_text\"].str.replace('\\n','').to_list())\n",
    "keep_ind = [i for i in range(len(pred[0])) if pred[0][i][0] == '__label__en' and pred[1][i][0] > .90]\n",
    "dat = dat.iloc[keep_ind,].reset_index(drop=True)\n",
    "print(\"english reviews: \", len(dat))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create days since added column\n",
    "review_dates = pd.to_datetime(pd.to_datetime(dat[\"date_added\"],format='%a %b %d %H:%M:%S %z %Y',errors='coerce'),utc=True,errors='coerce')\n",
    "dat[\"days_since_review\"] = (max(review_dates) - review_dates).dt.days\n",
    "\n",
    "# add user_reviews column\n",
    "dat = dat.merge(user_reviews,on='user_id')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final length:  1710198\n"
     ]
    }
   ],
   "source": [
    "# rename and reorder columns\n",
    "dat = dat.rename(columns={\"rating\": \"user_rating\", \"text_reviews_count\":\"book_reviews\", \"average_rating\":\"avg_rating\"})\n",
    "dat = dat[[\"user_id\", \"book_id\", \"user_reviews\", \"user_rating\", \"avg_rating\", \"ratings_count\", \"review_text\", \"days_since_review\", \"review_likes\", \"like_share\", \"popular\"]]\n",
    "print(\"final length: \", len(dat))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>user_reviews</th>\n",
       "      <th>user_rating</th>\n",
       "      <th>avg_rating</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>review_text</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>review_likes</th>\n",
       "      <th>like_share</th>\n",
       "      <th>popular</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>389241</th>\n",
       "      <td>d017f36b4a3844f25bfd9c0c0e20c0c1</td>\n",
       "      <td>33928870</td>\n",
       "      <td>240</td>\n",
       "      <td>5</td>\n",
       "      <td>4.32</td>\n",
       "      <td>1044</td>\n",
       "      <td>Should I be scared? Afraid? Happy? Excited? We...</td>\n",
       "      <td>221</td>\n",
       "      <td>2</td>\n",
       "      <td>0.002513</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>107419</th>\n",
       "      <td>b0349696a6e57285497085a8d16212f0</td>\n",
       "      <td>23899174</td>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "      <td>3.51</td>\n",
       "      <td>42060</td>\n",
       "      <td>A really great read!! What I like best about t...</td>\n",
       "      <td>413</td>\n",
       "      <td>1</td>\n",
       "      <td>0.002227</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1346016</th>\n",
       "      <td>e91c8245c5e65fae34ec3bf7fd9479f0</td>\n",
       "      <td>33016249</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>4.46</td>\n",
       "      <td>2988</td>\n",
       "      <td>I absolutely loved this book. Since I was youn...</td>\n",
       "      <td>144</td>\n",
       "      <td>1</td>\n",
       "      <td>0.002336</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>484216</th>\n",
       "      <td>93e91a7280ff388614923bcef2821dbe</td>\n",
       "      <td>21870140</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4.07</td>\n",
       "      <td>94</td>\n",
       "      <td>A very wild surreal romp lead by the fabulousn...</td>\n",
       "      <td>1132</td>\n",
       "      <td>1</td>\n",
       "      <td>0.006711</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>344458</th>\n",
       "      <td>a01f55c72e260953c8bb762dc96bf269</td>\n",
       "      <td>6931356</td>\n",
       "      <td>150</td>\n",
       "      <td>1</td>\n",
       "      <td>3.86</td>\n",
       "      <td>140645</td>\n",
       "      <td>I think YA has had quite a lot of girls who ma...</td>\n",
       "      <td>1202</td>\n",
       "      <td>12</td>\n",
       "      <td>0.002657</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                  user_id   book_id  user_reviews  \\\n",
       "389241   d017f36b4a3844f25bfd9c0c0e20c0c1  33928870           240   \n",
       "107419   b0349696a6e57285497085a8d16212f0  23899174            10   \n",
       "1346016  e91c8245c5e65fae34ec3bf7fd9479f0  33016249             1   \n",
       "484216   93e91a7280ff388614923bcef2821dbe  21870140             1   \n",
       "344458   a01f55c72e260953c8bb762dc96bf269   6931356           150   \n",
       "\n",
       "         user_rating avg_rating ratings_count  \\\n",
       "389241             5       4.32          1044   \n",
       "107419             5       3.51         42060   \n",
       "1346016            5       4.46          2988   \n",
       "484216             3       4.07            94   \n",
       "344458             1       3.86        140645   \n",
       "\n",
       "                                               review_text  days_since_review  \\\n",
       "389241   Should I be scared? Afraid? Happy? Excited? We...                221   \n",
       "107419   A really great read!! What I like best about t...                413   \n",
       "1346016  I absolutely loved this book. Since I was youn...                144   \n",
       "484216   A very wild surreal romp lead by the fabulousn...               1132   \n",
       "344458   I think YA has had quite a lot of girls who ma...               1202   \n",
       "\n",
       "         review_likes  like_share  popular  \n",
       "389241              2    0.002513        0  \n",
       "107419              1    0.002227        0  \n",
       "1346016             1    0.002336        0  \n",
       "484216              1    0.006711        0  \n",
       "344458             12    0.002657        0  "
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dat.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Jg4-imlKxyfI"
   },
   "source": [
    "##### Save filtered_reviews.csv in the current repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "id": "VES9t_zLxvzR"
   },
   "outputs": [],
   "source": [
    "save_path = '../data/filtered_reviews.csv'\n",
    "dat.to_csv(save_path, index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xPg43xGlvmv_"
   },
   "source": [
    "##### Peek at the filetered_reviews dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {
    "id": "j1wM7WR-vaRO"
   },
   "outputs": [],
   "source": [
    "filtered_reviews = pd.read_csv(save_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 17,
     "status": "ok",
     "timestamp": 1745699333259,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "J3AkDX3FyaeJ",
    "outputId": "f77c2d30-9f08-4b58-a73e-2a01c3e1e41d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1710198"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(filtered_reviews)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MyPBiYOgwyRB"
   },
   "source": [
    "Each row refers to a single review given by a user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 147
    },
    "executionInfo": {
     "elapsed": 211,
     "status": "ok",
     "timestamp": 1745698627229,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "uV2EhFCGvxC6",
    "outputId": "03423850-dcb0-4f2e-cc33-00f9ae622549"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>book_id</th>\n",
       "      <th>user_reviews</th>\n",
       "      <th>user_rating</th>\n",
       "      <th>avg_rating</th>\n",
       "      <th>ratings_count</th>\n",
       "      <th>review_text</th>\n",
       "      <th>days_since_review</th>\n",
       "      <th>review_likes</th>\n",
       "      <th>like_share</th>\n",
       "      <th>popular</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>18245960</td>\n",
       "      <td>218</td>\n",
       "      <td>5</td>\n",
       "      <td>4.01</td>\n",
       "      <td>6336</td>\n",
       "      <td>This is a special book. It started slow for ab...</td>\n",
       "      <td>96</td>\n",
       "      <td>29</td>\n",
       "      <td>0.145729</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8842281e1d1347389f2ab93d60773d4d</td>\n",
       "      <td>28684704</td>\n",
       "      <td>218</td>\n",
       "      <td>3</td>\n",
       "      <td>4.10</td>\n",
       "      <td>13990</td>\n",
       "      <td>A fun, fast paced science fiction thriller. I ...</td>\n",
       "      <td>353</td>\n",
       "      <td>22</td>\n",
       "      <td>0.021782</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                            user_id   book_id  user_reviews  user_rating  \\\n",
       "0  8842281e1d1347389f2ab93d60773d4d  18245960           218            5   \n",
       "1  8842281e1d1347389f2ab93d60773d4d  28684704           218            3   \n",
       "\n",
       "   avg_rating  ratings_count  \\\n",
       "0        4.01           6336   \n",
       "1        4.10          13990   \n",
       "\n",
       "                                         review_text  days_since_review  \\\n",
       "0  This is a special book. It started slow for ab...                 96   \n",
       "1  A fun, fast paced science fiction thriller. I ...                353   \n",
       "\n",
       "   review_likes  like_share  popular  \n",
       "0            29    0.145729        1  \n",
       "1            22    0.021782        1  "
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "filtered_reviews.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 140
    },
    "executionInfo": {
     "elapsed": 22,
     "status": "ok",
     "timestamp": 1745698949899,
     "user": {
      "displayName": "Gauranga Kumar Baishya",
      "userId": "09042457344042071897"
     },
     "user_tz": -330
    },
    "id": "g-7lt3U6vxZo",
    "outputId": "1264dab7-c8fd-4a3c-88fd-c1178b936499"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Two words: wesley ayers \\n I would say this book is actually around a 3.6ish rating instead of a full 4 stars. [only because it took me a while to get into] \\n There was a lot of information at the beginning and while I knew it was necessary, I still felt compelled to skim it over. [I\\'m so glad I didn\\'t lol] \\n Once the action started, it never stopped . There was a sense of urgency at every turn and Mac was on a mission. A deathly mission (excuse the pun). Mac\\'s mind is interesting. She\\'s in the middle of an intense grief and the way it influences her decisions, the way she talks, and acts was fascinating and I\\'m glad her grief was ingrained in every step she took, even if it was horribly awful to read about. \\n The second half was amazing, I couldn\\'t put it down and while, I predicted the \"major twist\" way beforehand, I thought how Schwab executed it was creepy as hell, but very fitting. \\n ANOTHER NOTE: wesley ayers is my fav'"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "filtered_reviews['review_text'][np.random.randint(low=0,high=1710198)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "like_share_bin\n",
      "(-0.00097, 0.05]    1594148\n",
      "(0.05, 0.1]           56517\n",
      "(0.1, 0.15]           20764\n",
      "(0.15, 0.2]           11025\n",
      "(0.2, 0.25]            6663\n",
      "(0.25, 0.3]            4475\n",
      "(0.3, 0.35]            3156\n",
      "(0.35, 0.4]            2414\n",
      "(0.4, 0.45]            1760\n",
      "(0.45, 0.5]            1453\n",
      "(0.5, 0.55]            1156\n",
      "(0.55, 0.6]             978\n",
      "(0.6, 0.65]             840\n",
      "(0.65, 0.7]             727\n",
      "(0.7, 0.75]             612\n",
      "(0.75, 0.8]             539\n",
      "(0.8, 0.85]             483\n",
      "(0.85, 0.9]             537\n",
      "(0.9, 0.95]             545\n",
      "(0.95, 1.0]            1406\n",
      "Name: count, dtype: int64\n"
     ]
    },
    {
     "data": {
      "image/png": "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SxcZO3asua8TsumY16pVq6bqPukK2+s8LTqxo15ld5XNdb5r166Q+9qVU4957TKuY131AomLbK7ztOjcHdqtc/369eIqm8+h5513nukWrsMhchIXjvOUQ620JVpbS/2NasgeOufPjz/+KD///LOZwwCnIWrzyjvO9rX1MkuXSNBllFykS+HYupTZyXTu3Nna9aNnzJjh5cuXzzv33HO9s846yyw/9M033wQet3n5OFePc10LuHDhwt51113nlStXzhs8eLCpd/130KBBZnm5N954w3ONzXWu612ntemyMrVq1Qrcd43Nde7qurou1/mzzz7rff311+a2ri+uSzoFL716xRVXeH/99ZfnGpvPoVq3+vdb1xlv27atN3HiRO/48eOe62w+znV5uIYNG5o612M6MTHRrEfvP9Y17pTL4rqgX79+Xrdu3Txb6XJl+jtR/5YEbxdffHFg2UrXdI7y3xa6u2eSjrXo1auXmcU4vauML774orlKjci1+OpEFTobJrLGRRddJC1btjQzueqfiuHDh5tWgU8++cQsFaKTa+hkIC5MDuYKbTF/4okn5JZbbjETVOlyQzqx4O23324e15lIdfyUTsyCrKHL2+i4+QsvvDCwT493PdZ1TJp/qSFXezHYSJdG1PF/WrfaY6FSpUrm2PZ74IEHZPny5aaVF1lD63jKlCnmPKmTgel8EVrn/pnp9VjX2Y61Jw+y7jjXuTm0FVdnitZJS7XHpU4Spn/Tte6RtbQnjvb609no33vvPdm6dav5G69jvfX76NatmxnCZPMqDGnRnkbaI+Obb74R2/zvf/8zOU+/fv3MTO466Z2uXKQrvnz44Yfy6aefmnHrNk+qmV4vHV0Cz9+rMdJI0jNJD0JNCs8555w0H9cTXv369eXIkSMRjy2nuu666076uK57rV0kSRizjq4t/ssvvwSWMlP6B/fOO+80s3frH2CS9Kylw2Z0vVT/knH6t0Z/TOuPZ6XdUrXe//rrryhHmnPojNz6A0iXHdJEXH/IKf1hp3/no7kES25I0vVvyKRJk0IukuhswC1atLByDWlX6d8S/W1SuXJlM0u0rsIQPAmV/p3RWbt1/XFk/XGu9Af/uHHjzI9+XWJL57zQGb1t7a7vIq3rmTNnmt/g+rtQL4rMnTtXmjVrZh7X3zQ6LNU/SRhOn/490cbJdu3amfu6JKI2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Xy7169eXzZs3yw8//BDy2JIlS+Sss84SV+kEMt9//73YpkuXLtKxY0fZunWrfPPNNyGP6XIV1apVE1dxnEcedR551HnkUeeRxzk08jjOI8/W4zwnX4yizsMXm4nXIIP0ymNCQoJZW88WGk+wIkWKhNzXGVQfffRRcdXYsWNl6NCh5uTyxRdfiA3SugIdrEyZMjJkyBBxFcd55FHnkUedRx51HnmcQyOP4zzybDzO/Rej/PRilK4xnlMuRlHn4WOddOQ4R44ckW+//Vbat28f7VAAAHAK51DkBq4d57oMXt68eaVt27biKuo8PCTpmbR48WJp2LChuGrHjh3y448/mn9V2bJl5YILLjD/IrIOHTpkjqdLL7002qHAIrpGbZ48eQL39f/r0aNHzVVePWnYytW409KtWzd55plnpHz58uISV+PWtWk3btwopUuXlri4OHGFq3G7at++ffLJJ5+Ybti6Xvr//d//WVvvrv5WdDVuICchSc+kmJgYqVq1qnTv3t2MJ3Llx5AmhP/9739l/PjxZpxF8eLFzf4///xTZ/qXm2++Wd544w0pVKiQuMbVZPfXX3+V888/3yQ3Nv747N+/v0yaNMkcKz179jTHvN/OnTvNsW9b7K7GrbZv325+dGo3q4svvlgmT54st912m3z55Zfm8Ro1asicOXOkXLlyYhNX41Y6OVJaGjVqJB9//LH5W6/q1q0rNnE1bjVs2DAzBlcnpdL/h9p19pVXXjGTUen5VY8dPRfZdmHH1bhdTnavu+46ueWWW+T666+XlStXSosWLczvFz2+9eKI3tZuqrVq1RLbuPpb0dW4/Xbt2iUrVqwwFxr0mNZz/jvvvGOGFnTo0EHOO+88cS3uK6+8UurUqSM208nX5s2bZ34P+I+h1q1bS9GiRcVmG2yNO2rzyueAaft79OhhlqaIjY31OnTo4H322WdeUlKSZ7Pbb7/dq1Gjhjd9+vSQWPX2jBkzvLPPPtu74447PBfpsiy65qFrbI574MCBXpkyZbzhw4d7/fv39+Li4rw777wz8PiOHTvM/wXbuBq3uu2227yLLrrImzJlinfjjTea25dccom3ZcsWb9OmTd7FF1/s3XPPPZ5tXI1b6bGg/wf135Sbf7+N/0ddjTvl+rT6//TMM88069SuXLnSe//998259bnnnvNs42rc6tprr/U++eQTc3vF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WlQaN25sdbwAAEQKSToAANmsbNmykj9//mx7/549e8qkSZPklVdekdWrV8v06dPl+uuvl71790p2Onbs2Gm9fuzYseYCxqJFi+Tiiy82MS9fvjzL4gMAwEUk6QAARLC7e0onTpyQ7t27S82aNWXz5s1mn7Yon3/++aZVvGrVqjJo0CBJSkpK9/2nTJkijz32mLRv3950a2/YsKHcd9995n1TtrjrvjPOOEPOOussefPNN0Mef/TRR+Xss8+WQoUKmc994okn5Pjx44HHn3zySalfv768/fbbUqVKFROf2rdvn9xxxx2mBV9bxS+77DL59ddfT1kvxYoVMxcw9DOffvppU8Zvv/028LhebGjWrJl5XokSJeTKK6+U33//PfC4f+iAXqBo2bKlibtevXqyYMGCkM956623pFKlSubxa6+9Vl544QXznsHCrXMAALILSToAAFFy9OhR+b//+z8zPv377783ibP+27lzZ+nVq5esWrVK3njjDRk3bpw888wz6b6PJrpffvmlHDhw4KSf9/zzz0ujRo1kyZIlcvfdd8tdd90la9asCTyuybt+ln7uSy+9ZJLbF198MeQ91q9fL59++qlJjP3j6rUMu3btkq+++koWL15skt3LL79c/vzzzwzVgybDo0ePNrfz5csX2H/o0CHp3bu3aWmfPXu2xMTEmCQ7OTk55PX9+/eXhx9+2MSjCf/NN98cSLB/+OEH09NA61Mfb926daq6zEydAwCQbTwAAHBaunTp4l1zzTXpPq6n288++8zcTkhIMPe///577/LLL/eaNWvm7du3L/Bc3ffss8+GvP69997zypUrl+77f/fdd17FihW9vHnzeo0aNfIeeOABb968eSHPqVy5snfrrbcG7icnJ3ulS5f2Xn/99XTfd/jw4V7Dhg0D9wcOHGg+Y9euXYF9Wo6iRYt6f//9d8hrq1Wr5r3xxhsnrZMCBQp4hQsX9mJiYsz9+Ph4b+/evem+Zvfu3eZ5y5cvD6nLt99+O/CclStXmn2//fabuX/jjTd6HTp0CHmfTp06eXFxcadV5wAAZBda0gEAiAJt7dWW4pkzZ0pcXFxgv3YTf+qpp6RIkSKBrUePHmbstnZXT8ull14qGzZsMK3NOq575cqVcskll5gu5MHq1q0buK3dxLUFXlvA/SZMmGDGhut+/dzHH3880AXfr3LlyqZbe3C8Bw8eNN3Rg2NOSEgI6ZqeFm2l19ZtbYGvXbu26UZfvHjxwOPr1q0z9aTdz7UbvXblVyljCi5XuXLlzL/+cmlPgSZNmoQ8P+X9zNQ5AADZJTbb3hkAAKRLx4+///77Zvy0juH204RXx0Nfd911qV7jHwOelrx585rEXDcdWz548GCTeOptfxdyfU4wTdT9Xcc1jk6dOpnPbtu2rblwMH78eNNFPljhwoVD7mu8mhjPmTMnVUwpx32npBcDqlevbjadRE7rRLubly5d2jx+1VVXmYsC2u2+fPnyJtY6deqkmrAuuFxaJpWyS/zJZLbOAQDIDiTpAABEgY4H14Tz6quvlmnTppkl05SO59bWX01cT4e2TOu47L///jtknHd65s+fbxJiHd/tt2nTplO+TuPdsWOHxMbGBlq6M0Nbt3XCOx0HruPhdWZ6rQdN0PXCg5o3b17Y73vOOefIzz//HLIv5f2sqnMAALICSToAAFkgMTExMJGan3YB11nF06MzsOvs7jpruXb51pnMBwwYYO7rJHLadV0nS9Pu2CtWrDCt42lp0aKF6Rauk8LpZ2prtM72rjOeazfxjKhRo4bpRq6t57pmuV44+Oyzz075ulatWknTpk3NOvHDhg0zE7dt27bNvF4nedOYMuqBBx4wr+nTp49pndey6Az0eltj69u3r4RL61iHA+iM7toy/80335i69re4q8zUOQAA2YUx6QAAZAHt7t2gQYOQTbtQZyQx1edpV29tzdau5lOnTjVj1TVZvvDCC83YbW3lTo++5p133pE2bdpIrVq1TGKq+z7++OMMx68t+g8++KDce++9Zpk1jUWXYDsVTXZ1ZnlNhLt162aS9Jtuusm0wpcpU0bCccUVV5il3bQ1XRNlvWCgs8VrjwONbfjw4RIuHWM/atQok6Tr8my6rJu+V3A39szUOQAA2cWns8dl27sDAABYRieFW716tVl6DQAA29DdHQAA5Gj/+9//zProOumddnXXXgevvfZatMMCACBNtKQDAIAc7YYbbjDDEQ4cOGCWc9PhAD179ox2WAAApIkkHQAAAAAASzBxHAAAAAAAliBJBwAAAADAEiTpAAAAAABYgiQdAAAAAABLkKQDAAAAAGAJknQAAAAAACxBkg4AAAAAgCVI0gEAAAAAEDv8P7CfjwPjsv2hAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Bin into 10 equal-width buckets\n",
    "filtered_reviews['like_share_bin'] = pd.cut(filtered_reviews['like_share'], bins=20)\n",
    "\n",
    "# Then see distribution\n",
    "print(filtered_reviews['like_share_bin'].value_counts().sort_index())\n",
    "\n",
    "# Or plot\n",
    "filtered_reviews['like_share_bin'].value_counts().sort_index().plot(kind='bar', figsize=(12,6))\n",
    "plt.title('Binned Distribution of Like Share')\n",
    "plt.xlabel('Like Share Range')\n",
    "plt.ylabel('Number of Reviews')\n",
    "plt.grid(True, axis='y', linestyle='--', alpha=0.7)\n",
    "plt.show()"
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
  "accelerator": "TPU",
  "colab": {
   "authorship_tag": "ABX9TyPhymh4OgFdEbZKbotO9knw",
   "gpuType": "V28",
   "mount_file_id": "1vgs77UGwAFno8Oy9wf0PU3Yqjbvh_Cmc",
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
