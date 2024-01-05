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

class DogBreedDetails(models.Model):
    type_choices = [(1,"Sporting Dog"), (2,"Hound Dog"), (3,"Working Dog"),
                                      (4,"Terrier Group"),(5,"Toy Group"),
                                      (6,"Non-Sporting Group"),
                                      (7,"Herding Group"),(8,"Miscellaneous Class"),
                                      (9,"Foundation Stock Service")]
    breed = models.CharField(max_length=50)
    description = models.TextField()
    character = models.CharField(max_length=50)
    height = models.CharField(max_length=20)
    weight = models.CharField(max_length=30)
    life_expentancy = models.CharField(max_length=30)
    aggressiveness = models.CharField(max_length=10)
    avatar = models.CharField(max_length=255)
    image0= models.CharField(max_length=255,null=True)
    image1= models.CharField(max_length=255,null=True)
    image2= models.CharField(max_length=255,null=True)
    # origin_id = models.ForeignKey(DogOrigin, on_delete=models.CASCADE,null=True)
    type_id = models.ForeignKey(DogType,on_delete=models.CASCADE)
    akc_link = models.CharField(max_length=255)
    

    
