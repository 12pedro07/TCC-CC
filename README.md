# Neonatal Face Mosaic: An areas-of-interest segmentation method based on 2D face images

## Publication

https://sol.sbc.org.br/index.php/wvc/article/view/18914

## Dependencies

- [Docker](https://www.docker.com);
- [docker-compose](https://docs.docker.com/compose/install);

## Running

1. Place your image dataset under `./Dataset`;
2. Edit the file `./src/config.yml`, replacing the parameter `input_dataset` with the name of your directory mentioned on step 1.
3. Edit any other necessary parameter on `./src/config.yml`;
4. Open your terminal or power shell;
5. Navigate to this directory;
6. Run:
    - if you have a nvidia gpu:
        ```
        docker-compose run MosaicPipeline-gpu
        ```
    - If you don't have one:
        ```
        docker-compose run MosaicPipeline-cpu
        ```

\* -d can be added to the command in order to run it in the background and the output can be checked with 
```
docker-compose logs -f
```

Note: On the first time running, docker will create the image, start the container and download the pre-trained neural network model, which may take a long time, but that is a one time process and won't be done again while the image is not deleted;

## Removing the docker container and image

1. From this directory run:
```
docker-compose down
```
2. After that run:
    - If you runned using the gpu:
        ```
        docker rmi mosaic-image-gpu
        ```
    - If you runned using the cpu:
        ```
        docker rmi mosaic-image-cpu
        ```

## Credits

[Insightface](https://github.com/deepinsight/insightface) - package used for face detection / key points regression
