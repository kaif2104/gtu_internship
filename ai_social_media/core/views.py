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

        follower_count = float(data.get("follower_count"))
        caption_length = float(data.get("caption_length"))
        hashtags_count = float(data.get("hashtags_count"))
        post_hour = float(data.get("post_hour"))
        has_call_to_action = int(data.get("has_call_to_action"))

        # Feature engineering
        engagement_score = caption_length + hashtags_count
        weighted_engagement = hashtags_count * 2
        engagement_per_follower = weighted_engagement / (follower_count + 1)
        log_followers = np.log1p(follower_count)
        caption_efficiency = caption_length / (hashtags_count + 1)
        is_peak = 1 if 18 <= post_hour <= 22 else 0
        is_weekend = 1

        input_dict = {
            "log_followers": log_followers,
            "caption_length": caption_length,
            "hashtags_count": hashtags_count,
            "post_hour": post_hour,
            "has_call_to_action": has_call_to_action,
            "engagement_score": engagement_score,
            "weighted_engagement": weighted_engagement,
            "engagement_per_follower": engagement_per_follower,
            "caption_efficiency": caption_efficiency,
            "is_peak": is_peak,
            "is_weekend": is_weekend
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        # Minimal refinement
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