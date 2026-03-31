pipeline {
    // 1. 'docker: not found' 에러를 방지하기 위해 docker가 설치된 agent를 지정합니다.
    // 만약 실습 환경에서 특정 label을 사용해야 한다면 'agent { label "docker-node" }' 등으로 수정하세요.
    agent any 

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        
        // 2. 반드시 Jenkins Credentials에 'harbor-robot-account'라는 ID로 
        // 로봇 계정(robot$skala26a-ai2 / Va9M8W...)을 등록해야 합니다.
        HARBOR_CREDS = 'harbor-robot-account' 
        
        BACKEND_IMAGE = 'sk047-myservice-backend'
        BACKEND_VER = '1.0.4'
        FRONTEND_IMAGE = 'sk047-myservice-frontend'
        FRONTEND_VER = '1.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                // SCM에서 코드 가져오기 (Jenkinsfile이 레포에 있으므로 checkout scm 사용 가능)
                checkout scm
            }
        }

        stage('Docker Build & Push') {
            steps {
                script {
                    // 3. Credentials에서 ID/PW를 안전하게 가져옵니다.
                    withCredentials([usernamePassword(credentialsId: "${HARBOR_CREDS}", 
                                                     usernameVariable: 'USER', 
                                                     passwordVariable: 'PASS')]) {
                        
                        // 4. Harbor 로그인 (계정명에 $가 포함되므로 '${USER}'와 같이 작은따옴표 권장)
                        echo "Logging into Harbor..."
                        sh "echo '${PASS}' | docker login ${HARBOR_URL} -u '${USER}' --password-stdin"

                        // 5. Backend 빌드 및 푸시
                        echo "Building Backend Image..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"

                        // 6. Frontend 빌드 및 푸시
                        echo "Building Frontend Image..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                    }
                }
            }
        }
    }

    post {
        always {
            // 7. 성공/실패 여부와 상관없이 로그아웃하여 세션 종료
            sh "docker logout ${HARBOR_URL} || true"
        }
        success {
            echo "Build and Push successfully completed!"
            // 로컬 이미지 삭제로 디스크 용량 정리
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
            sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
        }
    }
}