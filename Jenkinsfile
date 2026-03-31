pipeline {
    agent { label 'docker' } 

    tools {
        'org.jenkinsci.plugins.docker.commons.tools.DockerTool' 'jenkins-docker'
    }

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        // Jenkins Credentials에 저장된 'Username with password' ID여야 합니다.
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
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Deploy (Manual Login)'
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        // 1. Harbor 로그인 (플러그인 없이 sh로 직접 수행)
                        // HARBOR_CREDS_USR, HARBOR_CREDS_PSW는 credentials() 호출 시 자동 생성됩니다.
                        sh "echo ${HARBOR_CREDS_PSW} | docker login ${HARBOR_URL} -u ${HARBOR_CREDS_USR} --password-stdin"
                        
                        // 2. 이미지 푸시
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                        
                        // 3. 로그아웃 (보안)
                        sh "docker logout ${HARBOR_URL}"
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                try {
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
                    }
                } catch (e) {
                    echo "Cleanup skipped"
                }
            }
        }
    }
}