pipeline {
    // If 'any' fails, replace with the specific label of a node that has Docker
    // e.g., agent { label 'docker-enabled-node' }
    agent any 

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        HARBOR_CREDS = 'harbor-robot-account' 
        
        BACKEND_IMAGE = 'sk047-myservice-backend'
        BACKEND_VER = '1.0.4'
        FRONTEND_IMAGE = 'sk047-myservice-frontend'
        FRONTEND_VER = '1.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                echo '>>> Stage 1: Fetching source code'
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    echo '>>> Stage 2: Building images'
                    // Using sh only if the binary exists; otherwise this stage will fail
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Testing environment'
                sh "echo 'Testing connectivity to ${HARBOR_URL}'"
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Pushing to Harbor'
                    // This block requires the "Docker Pipeline" plugin to be installed in Jenkins
                    docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo 'SUCCESS: Pipeline finished'
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
        failure {
            echo 'FAILURE: Check if Docker is installed on the Jenkins Agent'
        }
    }
}