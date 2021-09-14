# Instalando dependencias

## Anaconda

1. Faça o download do anaconda
2. Abra o terminal, ou terminal do anaconda (caso use windows), neste diretório
3. Rode ``` conda env create -f environment.yml  ``` para criar e instalar um env com todos os requisitos

__Extras:__

- ``` conda activate tcc ``` para entrar no environment;
- ``` conda deactivate ``` para retornar ao environment base;
- ``` conda remove --name tcc --all ``` para remover o environment (desinstalar);
- ``` conda env list ``` para listar os envs que você tem criado;

## Pip

O pip não garante que a versão seja compativel, mas caso ja tenha um environment ou uma versão do python certa para utilizar:

1. Abra um cmd ou terminal nesta pasta;
2. Garanta que o python está na versão correta ``` python --version ```
3. Rode ``` pip install -r requirements.txt ``` para instalar as dependencias necessárias

# Configurando âmbiente de programação

## VSCode

- Caso utilize o anaconda: Na parte inferior esquerda, você pode encontrar "python" e a versão, clique para selecionar o environment do conda que deseja utilizar. Após isso deve estar escrito algo como: "Python 3.7.9 64-bit ('tcc': conda)", assim o terminal do vscode será capaz de rodar por conta própria os scripts já utilizando o ambiente correto.

- Extensões úteis:

1. Python
2. Jupyter
3. Git Graph
4. Docker

# Rodando o código

Para rodar, nós vamos criar uma imagem de um ubuntu 18, com tudo que é necessário para rodar e depois, vamos rodar essa imagem dentro de um container docker, desta forma o âmbiente fica totalmente instalado, qualquer problema é só excluir e remontar container. Algumas pastas serão montadas como volumes, para facilitar e agilizar o trabalho, ou seja, pastas como `src/`, `/Dataset` e `/Models` não serão copiados para dentro da imagem, no lugar, vamos fazer o mount delas dentro do container, dessa forma teremos as mesmas pastas dentro e fora, qualquer arquivo adicionado, modificado ou removido vai aparecer nos dois lugares.

## Criando a imagem

Para criar a imagem, navegue para a pasta deste arquivo (installation) e rode o seguinte comando (Observe que é preciso ter o docker-compose instalado)

```
docker-compose build
```

A imagem será criada seguindo as instruções nos arquivos docker-compose.yml e Dockerfile (pode demorar até algumas horas, mas só precisa ser feito uma vez na vida)

Para verificar a criação da imagem basta fazer `docker images`, você deve ver a nossa imagem (tcc-image) e a imagem base que foi utilizada (cuda).

Para excluir uma imagem use `docker rmi <imagem>`, onde a imagem pode ser os primeiros digitos do id ou o nome da imagem.

## Criando o container

Para iniciar o container use 

```
docker-compose up -d
```

A opção -d significa daemon, faz o container rodar em plano de fundo. 

Para ver os containers rodando use `docker ps` e para ver todos os containers rodando ou parados use `docker ps -a`. 

Para parar um container: `docker-compose down` ou `docker stop <container>`.

Para excluir um container: `docker rm <container>` com o container parado, ou adicione -f caso queira excluir com ele rodando.

## Uteis

O container vai rodar em plano de fundo, porem nos conseguimos acessa-lo pelo vscode usando a extensao do docker, tambem conseguimos abrir um terminal e entrar dentro dele enquanto estiver rodando utilizando `docker exec -it <container> bash` e conseguimos ver o output do que estiver rodando usando `docker-compose logs -f` (-f caso queira ficar vendo o output, se quiser so que imprima uma vez tire o -f).

## Links

1. [Insightface - Funções que importamos no python](https://github.com/deepinsight/insightface/tree/master/python-package/insightface)
2. [Insightface - Biblioteca completa](https://github.com/deepinsight/insightface)
3. [Docker - Opções de imagens base com cuda e cudnn](https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated&name=cudnn8-devel-ubuntu18)