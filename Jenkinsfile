pipeline {
    agent any

    environment {
        IMAGE_NAME = 'watermarking-app'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${IMAGE_NAME}")
                }
            }
        }

        stage('Push to DockerHub') {
        steps {
        script {
            sh 'echo S@i1319 | docker login -u Aashritha --password-stdin'
            sh "docker tag ${IMAGE_NAME} Aashritha/${IMAGE_NAME}"
            sh "docker push Aashritha/${IMAGE_NAME}"
        }
        }
    }
    }

    post {
        always {
            echo 'Pipeline execution completed.'
        }
    }
}