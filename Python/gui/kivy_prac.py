# Program to explain how to add image in kivy

# import kivy module
import kivy

# base Class of your App inherits from the App class.
# app:always refers to the instance of your application
from kivy.app import App
from kivy.uix.image import AsyncImage

# this restrict the kivy version i.e
# below this kivy version you cannot
# use the app or software
kivy.require('1.9.0')


# The Image widget is used to display an image
# this module contain all features of images


# creating the App class
class MyApp(App):

    # defining build()

    def build(self):
        # return image
        # return Image(source='download.jpeg')

        return AsyncImage(source='download.jpeg')

    # run the App


MyApp().run()
