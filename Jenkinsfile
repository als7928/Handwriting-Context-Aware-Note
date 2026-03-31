pipeline {
    // CHANGE 'any' to 'docker' or the specific label provided by your admin
    agent { label 'docker' } 

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
                echo '>>> Stage 1: Checkout'
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    echo '>>> Stage 2: Build'
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Test'
                sh "echo 'Testing images...'"
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy'
                    docker.withRegistry("https://${HARBOR_URL}", "${HARBOR_CREDS}") {
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Finalizing pipeline...'
        }
    }
}