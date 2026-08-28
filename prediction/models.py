from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=255, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Prediction: {self.prediction} at {self.uploaded_at}"
