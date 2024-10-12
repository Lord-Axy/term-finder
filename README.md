Term Finder app, to increase the dictionary add more terms and descriptions in csv file.

To run the gradio app, build docker and run it

commands:
docker build --no-cache -t termfinder . 
docker run --name mycontainer -p 8000:8000 termfinder
