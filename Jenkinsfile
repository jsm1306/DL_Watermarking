pipeline {
    agent any

    environment {
        IMAGE_NAME = 'watermarking-app'
        DOCKERHUB_USER = 'Aashritha'
    }

    stages {
        stage('Checkout Code') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                script {
                    docker.build("${IMAGE_NAME}")
                }
            }
        }

        stage('Push to DockerHub') {
            steps {
                echo 'Pushing image to DockerHub...'
                script {
                    // Use Jenkins credentials securely
                    withCredentials([usernamePassword(credentialsId: 'dockerhub-cred', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                        sh 'echo $PASS | docker login -u $USER --password-stdin'

                        // Tag image with build number (useful for versioning)
                        def VERSION = "v1.0.${env.BUILD_NUMBER}"
                        sh "docker tag ${IMAGE_NAME} ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
                        sh "docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"

                        // Optionally push a 'latest' tag too
                        sh "docker tag ${IMAGE_NAME} ${DOCKERHUB_USER}/${IMAGE_NAME}:latest"
                        sh "docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:latest"

                        // Cleanup local images to save space
                        sh "docker rmi ${IMAGE_NAME} ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION} ${DOCKERHUB_USER}/${IMAGE_NAME}:latest || true"
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'Image pushed successfully to DockerHub!'
        }
        failure {
            echo 'Pipeline failed! Check logs for details.'
        }
        always {
            echo 'Pipeline execution completed.'
        }
    }
}
