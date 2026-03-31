pipeline {
    agent any

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        HARBOR_CREDS = 'als7928_personal_access_token' // Jenkins Credentials ID
        
        // 서비스별 설정
        BACKEND_IMAGE = 'sk047-myservice-backend'
        BACKEND_VER = '1.0.4'
        FRONTEND_IMAGE = 'sk047-myservice-frontend'
        FRONTEND_VER = '1.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Docker Build & Push') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: "${HARBOR_CREDS}", 
                                                     usernameVariable: 'USER', 
                                                     passwordVariable: 'PASS')]) {
                        
                        // 1. Harbor 로그인
                        sh "echo ${PASS} | docker login ${HARBOR_URL} -u '${USER}' --password-stdin"

                        // 2. Backend 빌드 및 푸시
                        echo "Processing Backend..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"

                        // 3. Frontend 빌드 및 푸시
                        echo "Processing Frontend..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }

        stage('Cleanup') {
            steps {
                sh "docker logout ${HARBOR_URL}"
                // 빌드 서버 용량 확보를 위해 로컬 이미지 삭제 (선택 사항)
                sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
            }
        }
    }
}