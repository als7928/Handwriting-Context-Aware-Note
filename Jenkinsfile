pipeline {
    // 1. Ensure your Jenkins node has the label 'docker'
    agent { label 'docker' } 

    // 2. The name 'jenkins-docker' must match the 'Name' field in your 
    //    Jenkins Global Tool Configuration -> Docker section.
    tools {
        docker 'jenkins-docker' 
    }

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
                    // Building Backend
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                    // Building Frontend
                    sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                }
            }
        }

        stage('Test') {
            steps {
                echo '>>> Stage 3: Test'
                sh "docker --version"
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy (Push to Harbor)'
                    // This block requires 'Docker Pipeline' plugin and 'harbor-robot-account' credentials
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
            echo 'SUCCESS: Images pushed to Harbor.'
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
        failure {
            echo 'FAILURE: Check Docker tool name or Node permissions.'
        }
    }
}