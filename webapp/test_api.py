import requests
import json

# Use a video already in your Supabase
video_url = "https://gbishrvgqkvjdjovgrkw.supabase.co/storage/v1/object/public/videos/original/1754006477418_2025TbilisiFINALChoivBorodachev-00.01.31.217-00.01.33.066.mp4"

response = requests.post('http://localhost:5001/process', 
    json={
        'video_url': video_url,
        'video_id': 'test123'
    }
)

print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))