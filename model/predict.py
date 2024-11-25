import joblib
import pandas as pd

# # Load pipeline
# with open('model/model_pipeline.pkl', 'rb') as f:
#     loaded_pipeline = joblib.load(f)
    
# Predict new customer conversion rate  
new_customer = {'LeadID': 1,'Age':60,'Gender': 'Female','Location': 'Lahore',\
                'LeadSource': 'Organic','TimeSpentMinutes': 46,'PagesViewed': 6,\
                'LeadStatus': 'Hot','EmailSent': 10,'DeviceType': 'Mobile','ReferralSource':'Facebook',\
                'FormSubmissions': 2,'Downloads': 3,'CTR_ProductPage': 0.8,'ResponseTimeHours': 11,\
                'FollowUpEmails': 3,'SocialMediaEngagement': 54, 'PaymentHistory': 'Good'}

print(new_customer)
print('ici')