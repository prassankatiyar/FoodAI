# AI Food Tracker

## What is this?
I built a website that scans food using your webcam and tells you how many calories are in it. It works entirely locally and does not need the internet.

## How it Works
1.  **See:** You hold food (like an Apple, Burger, or Dosa) in front of the camera.
2.  **Think:** The Artificial Intelligence recognizes what the food is.
3.  **Result:** The app looks up the nutrition info from a local file and adds it to your daily log.

**Accuracy:** The model is about 80% accurate.

## The AI Logic
I used a technique called **Transfer Learning**.
* Instead of building a computer brain from zero, I used a pre-made smart model called **MobileNetV2**.
* I "retrained" this model to specifically recognize the foods in my folder.
* I made the training harder by rotating and zooming the images, so the AI learns to recognize food from different angles.
x

## Tools I Used
* **Python:** The main programming language.
* **Flask:** To run the website.
* **TensorFlow:** To run the AI.
* **HTML/JavaScript:** For the screen you see and the webcam.

## How to Run It
1.  **Install:** Run `pip install tensorflow flask pillow numpy`
2.  **Train:** Run `python train_model.py` to teach the AI.
3.  **Run:** Run `python app.py` to start the website.
4.  **Open:** Go to `http://127.0.0.1:5000` in your browser.