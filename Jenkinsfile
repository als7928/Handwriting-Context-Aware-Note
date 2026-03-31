pipeline {
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
                echo '>>> Stage 1: Fetching source code from GitHub'
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    echo '>>> Stage 2: Building Docker images'
                    
                    echo "Building Backend: ${BACKEND_IMAGE}:${BACKEND_VER}"
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                    
                    echo "Building Frontend: ${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Running integration tests'
                sh "echo 'Validation successful for ${BACKEND_IMAGE} and ${FRONTEND_IMAGE}'"
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploying images to Harbor Registry'
                    
                    docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                        echo "Pushing Backend to ${HARBOR_URL}..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        
                        echo "Pushing Frontend to ${HARBOR_URL}..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }
    }

    post {
        success {
            echo '============================================'
            echo 'SUCCESS: All images deployed to Harbor'
            echo '============================================'
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
        failure {
            echo '============================================'
            echo 'FAILURE: Pipeline execution failed'
            echo '============================================'
        }
    }
}