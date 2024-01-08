from django.db import models

# Create your models here.
class DogOrigin(models.Model):
    origin_place = models.CharField(null=True,max_length=30)

class DogType(models.Model):
    type_choices = [("Sporting Dog","Sporting Dog"), ("Hound Dog","Hound Dog"), ("Working Dog","Working Dog"),
                                      ("Terrier Group","Terrier Group"),("Toy Group","Toy Group"),
                                      ("Non-Sporting Group","Non-Sporting Group"),
                                      ("Herding Group","Herding Group"),("Miscellaneous Class","Miscellaneous Class"),
                                      ("Foundation Stock Service","Foundation Stock Service")]
    type = models.CharField(max_length=50,choices=type_choices)
    def __str__(self):
        return f"{self.type}"

class DogBreedDetails(models.Model):
    breed = models.CharField(max_length=50)
    description = models.TextField()
    character = models.CharField(max_length=50)
    height = models.CharField(max_length=20)
    weight = models.CharField(max_length=30)
    life_expentancy = models.CharField(max_length=30)
    aggressiveness = models.CharField(max_length=50)
    avatar = models.CharField(max_length=255)
    image0= models.CharField(max_length=255,null=True)
    image1= models.CharField(max_length=255,null=True)
    image2= models.CharField(max_length=255,null=True)
    type_id = models.ForeignKey(DogType,on_delete=models.CASCADE,)
    akc_link = models.CharField(max_length=255)
    

    
