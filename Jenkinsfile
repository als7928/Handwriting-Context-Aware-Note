pipeline {
    agent { label 'docker' } 

    tools {
        // 기존에 작동 확인된 도구 설정 유지
        'org.jenkinsci.plugins.docker.commons.tools.DockerTool' 'jenkins-docker'
    }

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        
        // Jenkins Credentials에 등록된 ID가 'harbor-robot-account'인지 확인하세요.
        // credentials() 함수를 사용하면 _USR, _PSW 변수가 자동으로 생성됩니다.
        HARBOR_CREDS = credentials('harbor-robot-account') 
        
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

        stage('Build') {
            steps {
                script {
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        echo ">>> Building Images..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Logging in to Harbor and Pushing'
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        // 1. Docker Login (가장 확실한 stdin 방식 사용)
                        // echo 뒤의 변수는 credentials()에 의해 생성된 _PSW, _USR 입니다.
                        sh "echo ${HARBOR_CREDS_PSW} | docker login ${HARBOR_URL} -u '${HARBOR_CREDS_USR}' --password-stdin"
                        
                        // 2. Push Images
                        echo "Pushing Backend..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        
                        echo "Pushing Frontend..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                        
                        // 3. Logout
                        sh "docker logout ${HARBOR_URL}"
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                // 빌드 후 로컬 이미지 정리 (성공/실패 상관없이 실행)
                try {
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
                    }
                } catch (e) {
                    echo "Cleanup cleanup skipped."
                }
            }
        }
    }
}