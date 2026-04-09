import datetime
import pickle
import os
import numpy as np
import pandas as pd
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI


# 🔹 Load ML model
model_path = os.path.join(os.path.dirname(__file__), "final_social_media_model.pkl")
model = pickle.load(open(model_path, "rb"))

# 🔹 Load columns
columns_path = os.path.join(os.path.dirname(__file__), "model_columns.pkl")
model_columns = pickle.load(open(columns_path, "rb"))

# 🔹 OpenAI client
client = OpenAI(api_key="Your API KEY")


# 🔹 Home
def home(request):
    return render(request, "index.html")


# 🔹 ML Prediction
@csrf_exempt
@api_view(['POST'])
def predict_performance(request):
    try:
        data = request.data

        follower_count     = float(data.get("follower_count"))
        caption_length     = float(data.get("caption_length"))
        hashtags_count     = float(data.get("hashtags_count"))
        post_hour          = float(data.get("post_hour"))
        has_call_to_action = int(data.get("has_call_to_action"))
        media_type         = str(data.get("media_type", "reel")).lower()
        account_type       = str(data.get("account_type", "creator")).lower()
        content_category   = str(data.get("content_category", "Fitness"))
        day_of_week        = str(data.get("day_of_week", "Monday"))

        # Feature engineering (mirrors ML notebook)
        engagement_score       = caption_length + hashtags_count
        weighted_engagement    = hashtags_count * 2
        engagement_per_follower = weighted_engagement / (follower_count + 1)
        log_followers          = np.log1p(follower_count)
        caption_efficiency     = caption_length / (hashtags_count + 1)
        is_peak                = 1 if 18 <= post_hour <= 22 else 0

        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        if day_of_week not in valid_days:
            day_of_week = "Monday"
        is_weekend = 1 if day_of_week in ("Saturday", "Sunday") else 0

        # Additional features used in training
        caption_per_follower = caption_length / (log_followers + 1)
        balanced_score       = caption_length * 0.3 + hashtags_count * 2 + is_peak * 5
        hashtag_efficiency   = hashtags_count / (caption_length + 1)

        # Build base input dict (numeric/boolean features)
        input_dict = {
            "log_followers":          log_followers,
            "caption_length":         caption_length,
            "hashtags_count":         hashtags_count,
            "post_hour":              post_hour,
            "has_call_to_action":     has_call_to_action,
            "engagement_score":       engagement_score,
            "weighted_engagement":    weighted_engagement,
            "engagement_per_follower": engagement_per_follower,
            "caption_efficiency":     caption_efficiency,
            "is_peak":                is_peak,
            "is_weekend":             is_weekend,
            "caption_per_follower":   caption_per_follower,
            "balanced_score":         balanced_score,
            "hashtag_efficiency":     hashtag_efficiency,
            # One-hot encoded categorical columns (drop_first=True was used)
            f"account_type_{account_type}":         1,
            f"media_type_{media_type}":             1,
            f"content_category_{content_category}": 1,
            f"day_of_week_{day_of_week}":           1,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        # Heuristic refinement for edge cases
        if prediction == "medium":
            if follower_count < 2000:
                prediction = "low"
            elif follower_count > 30000:
                prediction = "high"

        return Response({"prediction": prediction})

    except Exception as e:
        return Response({"error": str(e)})


# 🔥 AI Caption Generator (FIXED)
@csrf_exempt
@api_view(['POST'])
def ai_caption(request):
    try:
        category = request.data.get("category")

        prompt = f"Generate an engaging Instagram caption for {category}."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return Response({
            "caption": response.choices[0].message.content
        })

    except Exception as e:
        return Response({
            "error": str(e)
        })


# 🔥 AI Hashtag Generator (FIXED)
@csrf_exempt
@api_view(['POST'])
def ai_hashtags(request):
    try:
        category = request.data.get("category")

        prompt = f"Give 10 trending Instagram hashtags for {category}."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return Response({
            "hashtags": response.choices[0].message.content
        })

    except Exception as e:
        return Response({
            "error": str(e)
        })