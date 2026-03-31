pipeline {
    agent any

    environment {
        // Harbor 설정
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        IMAGE_NAME = 'sk047-myservice-backend'
        VERSION = '1.0.4'
        
        // Jenkins에 등록한 Credentials ID
        HARBOR_CREDS = 'als7928_personal_access_token' 
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Docker Build') {
            steps {
                echo "Building Docker Image: ${IMAGE_NAME}:${VERSION}"
                // 로컬 빌드 및 Harbor 태그 생성
                sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${VERSION} ."
            }
        }

        stage('Harbor Login & Push') {
            steps {
                // withCredentials를 사용하여 안전하게 로그인 및 푸시
                withCredentials([usernamePassword(credentialsId: "${HARBOR_CREDS}", 
                                                 usernameVariable: 'USER', 
                                                 passwordVariable: 'PASS')]) {
                    echo "Logging into Harbor..."
                    sh "echo ${PASS} | docker login ${HARBOR_URL} -u '${USER}' --password-stdin"
                    
                    echo "Pushing Image to Harbor..."
                    sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${VERSION}"
                }
            }
        }

        stage('Cleanup') {
            steps {
                echo "Cleaning up local images..."
                sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${IMAGE_NAME}:${VERSION}"
                sh "docker logout ${HARBOR_URL}"
            }
        }
    }
}